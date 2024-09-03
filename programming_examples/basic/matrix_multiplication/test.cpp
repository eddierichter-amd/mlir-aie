//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <ctime>
#include <cstdint>
#include <sstream>
#include <string>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdfloat>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "common.h"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"


#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
#ifndef DTYPE_IN
#define DTYPE_IN std::bfloat16_t
#endif
#ifndef DTYPE_OUT
#define DTYPE_OUT std::bfloat16_t
#endif
#ifndef DTYPE_ACC
#define DTYPE_ACC float
#endif
using A_DATATYPE = DTYPE_IN;
using B_DATATYPE = DTYPE_IN;
using C_DATATYPE = DTYPE_OUT;
using ACC_DATATYPE = DTYPE_ACC;
#endif

#define XSTR(X) STR(X)
#define STR(X) #X

constexpr long long verify_stochastic_threshold = 1024 * 1024 * 1024;
constexpr int verify_stochastic_n_samples = 1000;

// Verification tolerance
// See "Note on Numerical Tolerances" in README.md
float abs_tol = matmul_common::get_abs_tol<C_DATATYPE>();
float rel_tol = matmul_common::get_rel_tol<C_DATATYPE>();

namespace po = boost::program_options;

namespace {

hsa_status_t get_agent(hsa_agent_t agent, std::vector<hsa_agent_t> *agents,
                       hsa_device_type_t requested_dev_type)
{
  if (!agents || !(requested_dev_type == HSA_DEVICE_TYPE_AIE ||
      requested_dev_type == HSA_DEVICE_TYPE_GPU)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_device_type_t device_type;
  hsa_status_t ret = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);

  if (ret != HSA_STATUS_SUCCESS) {
    return ret;
  }

  if (device_type == requested_dev_type) {
    agents->push_back(agent);
  }

  return ret;
}

hsa_status_t get_aie_agents(hsa_agent_t agent, void *data)
{
  if (!data) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto* aie_agents = reinterpret_cast<std::vector<hsa_agent_t>*>(data);
  return get_agent(agent, aie_agents, HSA_DEVICE_TYPE_AIE);
}

hsa_status_t get_gpu_agents(hsa_agent_t agent, void *data)
{
  if (!data) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto* gpu_agents = reinterpret_cast<std::vector<hsa_agent_t>*>(data);
  return get_agent(agent, gpu_agents, HSA_DEVICE_TYPE_GPU);
}

hsa_status_t get_coarse_global_mem_pool(hsa_amd_memory_pool_t pool, void *data, bool kernarg)
{
  hsa_amd_segment_t segment_type;
  auto ret = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                          &segment_type);
  if (ret != HSA_STATUS_SUCCESS) {
    return ret;
  }

  if (segment_type == HSA_AMD_SEGMENT_GLOBAL) {
    hsa_amd_memory_pool_global_flag_t global_pool_flags;
    ret = hsa_amd_memory_pool_get_info(pool,
                                       HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                       &global_pool_flags);
    if (ret != HSA_STATUS_SUCCESS) {
      return ret;
    }

    if (kernarg) {
      if ((global_pool_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) &&
          (global_pool_flags & HSA_REGION_GLOBAL_FLAG_KERNARG)) {
        *static_cast<hsa_amd_memory_pool_t*>(data) = pool;
      }
    } else {
      if ((global_pool_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) &&
          !(global_pool_flags & HSA_REGION_GLOBAL_FLAG_KERNARG)) {
        *static_cast<hsa_amd_memory_pool_t*>(data) = pool;
      }
    }
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t get_coarse_global_dev_mem_pool(hsa_amd_memory_pool_t pool, void *data)
{
  return get_coarse_global_mem_pool(pool, data, false);
}

hsa_status_t get_coarse_global_kernarg_mem_pool(hsa_amd_memory_pool_t pool, void *data)
{
  return get_coarse_global_mem_pool(pool, data, true);
}

void load_pdi_file(hsa_amd_memory_pool_t mem_pool, const std::string& file_name, void** buf)
{
  std::ifstream bin_file(file_name, std::ios::binary | std::ios::ate | std::ios::in);

  //assert_FALSE(bin_file.fail());

  auto size(bin_file.tellg());

  bin_file.seekg(0, std::ios::beg);
  assert(hsa_amd_memory_pool_allocate(mem_pool, size, 0, buf) == HSA_STATUS_SUCCESS);
  bin_file.read(reinterpret_cast<char*>(*buf), size);
}

void load_instr_file(hsa_amd_memory_pool_t mem_pool, const std::string& file_name, void** buf, uint32_t &num_instr)
{
  std::ifstream bin_file(file_name, std::ios::binary | std::ios::ate | std::ios::in);

  //assert_FALSE(bin_file.fail());

  auto size(bin_file.tellg());
  bin_file.seekg(0, std::ios::beg);
  std::vector<uint32_t> pdi_vec;
  std::string val;

  while (bin_file >> val) {
    pdi_vec.push_back(std::stoul(val, nullptr, 16));
  }

  assert(hsa_amd_memory_pool_allocate(mem_pool, size, 0, buf) == HSA_STATUS_SUCCESS);
  std::memcpy(*buf, pdi_vec.data(), pdi_vec.size() * sizeof(uint32_t));
  num_instr = pdi_vec.size();
}

} // namespace

namespace po = boost::program_options;

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  po::options_description desc("Allowed options");
  po::variables_map vm;
  matmul_common::add_default_options(desc);

  matmul_common::parse_options(argc, argv, desc, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();

  srand(time(NULL));

  int M = vm["M"].as<int>();
  int K = vm["K"].as<int>();
  int N = vm["N"].as<int>();
  bool do_verify_stochastic =
      (long long)M * N * K > verify_stochastic_threshold;

  if (verbosity >= 1) {
    std::cout << "Matrix size " << M << "x" << K << "x" << N << std::endl;
  }

  int A_VOLUME = M * K;
  int B_VOLUME = N * K;
  int C_VOLUME = M * N;

  size_t A_SIZE = (A_VOLUME * sizeof(A_DATATYPE));
  size_t B_SIZE = (B_VOLUME * sizeof(B_DATATYPE));
  size_t C_SIZE = (C_VOLUME * sizeof(C_DATATYPE));

  size_t OUT_SIZE = C_SIZE + trace_size;

  // List of AIE agents in the system.
  std::vector<hsa_agent_t> aie_agents;
  // For creating a queue on an AIE agent.
  hsa_queue_t *aie_queue(nullptr);
  // Memory pool for allocating device-mapped memory. Used for PDI/DPU instructions.
  hsa_amd_memory_pool_t global_dev_mem_pool{0};
  // System memory pool. Used for allocating kernel argument data.
  hsa_amd_memory_pool_t global_kernarg_mem_pool{0};
  const std::string instr_inst_file_name(vm["instr"].as<std::string>());
  const std::string pdi_file_name(vm["xclbin"].as<std::string>()); // Reusing xclbin argument as pdi
  uint32_t *instr_inst_buf(nullptr);
  uint64_t *pdi_buf(nullptr);

  assert(aie_agents.empty() && "No aie agents");
  assert(global_dev_mem_pool.handle == 0);
  assert(global_kernarg_mem_pool.handle == 0);

  // Initialize the runtime.
  assert(hsa_init() == HSA_STATUS_SUCCESS);

  assert(sizeof(hsa_kernel_dispatch_packet_s) == sizeof(hsa_amd_aie_ert_packet_s));

  // Find the AIE agents in the system.
  assert(hsa_iterate_agents(get_aie_agents, &aie_agents) == HSA_STATUS_SUCCESS);
  assert(aie_agents.size() == 1);

  const auto& aie_agent = aie_agents.front();

  // Create a queue on the first agent.
  assert(hsa_queue_create(aie_agent, 64, HSA_QUEUE_TYPE_SINGLE,
                           nullptr, nullptr, 0, 0, &aie_queue) == HSA_STATUS_SUCCESS);
  assert(aie_queue);
  assert(aie_queue->base_address);


  // Find a pool for DEV BOs. This is a global system memory pool that is mapped
  // to the device. Will be used for PDIs and DPU instructions.
  assert(hsa_amd_agent_iterate_memory_pools(aie_agent,
                                             get_coarse_global_dev_mem_pool,
                                             &global_dev_mem_pool) == HSA_STATUS_SUCCESS);

  // Find a pool that supports kernel args. This is just normal system memory. It will
  // be used for commands and input data.
  assert(hsa_amd_agent_iterate_memory_pools(aie_agent,
                                             get_coarse_global_kernarg_mem_pool,
                                             &global_kernarg_mem_pool) == HSA_STATUS_SUCCESS);

  // Load the DPU and PDI files into a global pool that doesn't support kernel args (DEV BO).
  uint32_t num_instr;
  load_instr_file(global_dev_mem_pool, instr_inst_file_name, reinterpret_cast<void**>(&instr_inst_buf), num_instr);
  uint32_t instr_handle = 0;
  assert(hsa_amd_get_handle_from_vaddr(instr_inst_buf, &instr_handle) == HSA_STATUS_SUCCESS);
  assert(instr_handle != 0);

  load_pdi_file(global_dev_mem_pool, pdi_file_name, reinterpret_cast<void**>(&pdi_buf));
  uint32_t pdi_handle = 0;
  assert(hsa_amd_get_handle_from_vaddr(pdi_buf, &pdi_handle) == HSA_STATUS_SUCCESS);
  assert(pdi_handle != 0);

  hsa_amd_aie_ert_hw_ctx_cu_config_t cu_config {
    .cu_config_bo = pdi_handle,
    .cu_func = 0
  };

  hsa_amd_aie_ert_hw_ctx_config_cu_param_t config_cu_args {
    .num_cus = 1,
    .cu_configs = &cu_config
  };

  // Configure the queue's hardware context.
  assert(hsa_amd_queue_hw_ctx_config(aie_queue, HSA_AMD_QUEUE_AIE_ERT_HW_CXT_CONFIG_CU,
                                      &config_cu_args) == HSA_STATUS_SUCCESS
  );

  // create inputs / outputs
  constexpr std::size_t num_data_elements = 256;
  constexpr std::size_t data_buffer_size = num_data_elements * sizeof(std::uint32_t);

  A_DATATYPE* input_a = {};
  assert(hsa_amd_memory_pool_allocate(global_dev_mem_pool,
                                       A_SIZE,
                                       0,
                                       reinterpret_cast<void**>(&input_a)) == HSA_STATUS_SUCCESS);
  std::uint32_t input_a_handle = {};
  assert(hsa_amd_get_handle_from_vaddr(input_a, &input_a_handle) == HSA_STATUS_SUCCESS);
  assert(input_a_handle != 0);


  B_DATATYPE* input_b = {};
  assert(hsa_amd_memory_pool_allocate(global_dev_mem_pool,
                                       B_SIZE,
                                       0,
                                       reinterpret_cast<void**>(&input_b)) == HSA_STATUS_SUCCESS);
  std::uint32_t input_b_handle = {};
  assert(hsa_amd_get_handle_from_vaddr(input_b, &input_b_handle) == HSA_STATUS_SUCCESS);
  assert(input_b_handle != 0);

  C_DATATYPE* output = {};
  assert(hsa_amd_memory_pool_allocate(global_dev_mem_pool,
                                       OUT_SIZE,
                                       0,
                                       reinterpret_cast<void**>(&output)) == HSA_STATUS_SUCCESS);
  std::uint32_t output_handle = {};
  assert(hsa_amd_get_handle_from_vaddr(output, &output_handle) == HSA_STATUS_SUCCESS);
  assert(output_handle != 0);

  if (verbosity >= 1) {
    std::cout << "Writing data into buffer objects.\n";
  }

  //A_DATATYPE *bufA = bo_a.map<A_DATATYPE *>();
  // Writing A
  std::vector<A_DATATYPE> AVec(A_VOLUME);
  for (int i = 0; i < A_VOLUME; i++) {
    AVec[i] = matmul_common::get_random<A_DATATYPE>();
    // AVec[i] = i;
  }
  memcpy(input_a, AVec.data(), (AVec.size() * sizeof(A_DATATYPE)));

  // Writing B
  //B_DATATYPE *bufB = bo_b.map<B_DATATYPE *>();
  std::vector<B_DATATYPE> BVec(B_VOLUME);
  for (int i = 0; i < B_VOLUME; i++) {
    BVec[i] = matmul_common::get_random<B_DATATYPE>();
  }
  memcpy(input_b, BVec.data(), (BVec.size() * sizeof(B_DATATYPE)));

  //char *bufOut = bo_out.map<char *>();
  std::vector<C_DATATYPE> CVec(C_VOLUME);
  memset(output, 0, OUT_SIZE);

  //for (std::size_t i = 0; i < A_VOLUME; i++) {
  //  *(input_a + i) = matmul_common::get_random<A_DATATYPE>();
  //}

  //for (std::size_t i = 0; i < B_VOLUME; i++) {
  //  *(input_b + i) = matmul_common::get_random<B_DATATYPE>();
  //}

  //for (std::size_t i = 0; i < C_VOLUME; i++) {
  //  *(output + i) = 0;
  //}

  if (verbosity >= 2) {
    std::cout << "DTYPE_IN  = " XSTR(DTYPE_IN) "\n";
    std::cout << "DTYPE_OUT = " XSTR(DTYPE_OUT) "\n";
    std::cout << "Verification tolerance " << abs_tol << " absolute, "
              << rel_tol << " relative.\n";
    std::cout << "A = \n";
    matmul_common::print_matrix(AVec, K);
    std::cout << "B = \n";
    matmul_common::print_matrix(BVec, N);
  }

  ///////////////////////////////////// Creating the cmd packet
  // Creating a packet to store the command
  hsa_amd_aie_ert_packet_t *cmd_pkt = NULL;
  assert(hsa_amd_memory_pool_allocate(global_kernarg_mem_pool, 64, 0, reinterpret_cast<void**>(&cmd_pkt)) == HSA_STATUS_SUCCESS);
  cmd_pkt->state = HSA_AMD_AIE_ERT_STATE_NEW;
  cmd_pkt->count = 0xC; // # of arguments to put in command
  cmd_pkt->opcode = HSA_AMD_AIE_ERT_START_CU; 
  cmd_pkt->header.AmdFormat = HSA_AMD_PACKET_TYPE_AIE_ERT; 
  cmd_pkt->header.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;

  // Creating the payload for the packet
  hsa_amd_aie_ert_start_kernel_data_t *cmd_payload = NULL;
  uint32_t cmd_handle;
  assert(hsa_amd_get_handle_from_vaddr(reinterpret_cast<void*>(cmd_pkt), &cmd_handle) == HSA_STATUS_SUCCESS);
  assert(hsa_amd_memory_pool_allocate(global_kernarg_mem_pool, 64, 0, reinterpret_cast<void**>(&cmd_payload)) == HSA_STATUS_SUCCESS);
  cmd_payload->cu_mask = 0x1; // Selecting the PDI to use with this command
  cmd_payload->data[0] = 0x3; // Transaction opcode
  cmd_payload->data[1] = 0x0;
  cmd_payload->data[2] = instr_handle; 
  cmd_payload->data[3] = 0x0;
  cmd_payload->data[4] = num_instr;
  cmd_payload->data[5] = input_a_handle;
  cmd_payload->data[6] = 0;
  cmd_payload->data[7] = input_b_handle;
  cmd_payload->data[8] = 0;
  cmd_payload->data[9] = output_handle;
  cmd_payload->data[10] = 0;
  cmd_pkt->payload_data = reinterpret_cast<uint64_t>(cmd_payload);

  uint64_t wr_idx = hsa_queue_add_write_index_relaxed(aie_queue, 1);
  uint64_t packet_id = wr_idx % aie_queue->size;
  reinterpret_cast<hsa_amd_aie_ert_packet_t *>(aie_queue->base_address)[packet_id] = *cmd_pkt;
  hsa_signal_store_screlease(aie_queue->doorbell_signal, wr_idx);

  // Copy the output into the vector
  memcpy(CVec.data(), output, (CVec.size() * sizeof(C_DATATYPE)));

  int errors = 0;
  if (do_verify) {
    if (verbosity >= 1) {
      if (do_verify_stochastic) {
        std::cout << "Verifying " << verify_stochastic_n_samples
                  << " random samples against reference matmul ..."
                  << std::endl;
      } else {
        std::cout << "Verifying against reference matmul ..." << std::endl;
      }
    }
    auto vstart = std::chrono::system_clock::now();
    if (do_verify_stochastic) {
      errors = matmul_common::verify_stochastic<A_DATATYPE, C_DATATYPE,
                                                ACC_DATATYPE>(
          M, N, K, AVec, BVec, CVec, verify_stochastic_n_samples, verbosity,
          abs_tol, rel_tol);
    } else {
      errors = matmul_common::verify<A_DATATYPE, C_DATATYPE, ACC_DATATYPE>(
          M, N, K, AVec, BVec, CVec, abs_tol, rel_tol);
    }
    auto vstop = std::chrono::system_clock::now();
    float vtime =
        std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart)
            .count();
    if (verbosity >= 1) {
      std::cout << "Verify time: " << vtime << " s." << std::endl;
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nError count: " << errors;
    if (do_verify_stochastic) {
      std::cout << " (out of " << verify_stochastic_n_samples
                << " random samples)";
    }
    std::cout << "\n\n";

    std::cout << "\nFailed.\n\n";
    return 1;
  }

  assert(hsa_queue_destroy(aie_queue) == HSA_STATUS_SUCCESS);

  assert(hsa_amd_memory_pool_free(output) == HSA_STATUS_SUCCESS);
  assert(hsa_amd_memory_pool_free(input_a) == HSA_STATUS_SUCCESS);
  assert(hsa_amd_memory_pool_free(input_b) == HSA_STATUS_SUCCESS);
  assert(hsa_amd_memory_pool_free(pdi_buf) == HSA_STATUS_SUCCESS);
  assert(hsa_amd_memory_pool_free(instr_inst_buf) == HSA_STATUS_SUCCESS);

  assert(hsa_shut_down() == HSA_STATUS_SUCCESS);
}
