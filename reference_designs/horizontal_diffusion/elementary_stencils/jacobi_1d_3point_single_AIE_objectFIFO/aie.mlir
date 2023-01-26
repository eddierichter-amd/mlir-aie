// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET     
//
//===----------------------------------------------------------------------===//

module @stencil_1d_3point{
  %t73 = AIE.tile(7, 3)
  // %t72 = AIE.tile(7, 2)
  // %t71 = AIE.tile(7, 1)
  %t70 = AIE.tile(7, 0)
  
  %lock73_14 = AIE.lock(%t73, 14) { sym_name = "lock73_14" }

  %obj_fifo_in = AIE.objectFifo.createObjectFifo(%t70, {%t73}, 2) {sym_name = "obj_in" }: !AIE.objectFifo<memref<256xi32>>
  %obj_fifo_out = AIE.objectFifo.createObjectFifo(%t73, {%t70}, 2){sym_name = "obj_out" } : !AIE.objectFifo<memref<256xi32>>

   // DDR buffer
  %ext_buffer_in0  = AIE.external_buffer  {sym_name = "ddr_test_buffer_in0"}: memref<256 x i32>
  %ext_buffer_out = AIE.external_buffer  {sym_name = "ddr_test_buffer_out"}: memref<256 x i32>
      
  // Register the external memory pointers to the object FIFOs.
  AIE.objectFifo.registerExternalBuffers(%t70, %obj_fifo_in : !AIE.objectFifo<memref<256xi32>>, {%ext_buffer_in0}) : (memref<256xi32>)
  AIE.objectFifo.registerExternalBuffers(%t70, %obj_fifo_out : !AIE.objectFifo<memref<256xi32>>, {%ext_buffer_out}) : (memref<256xi32>)

  func.func private @stencil_1d_3point(%A: memref<256xi32>,%B: memref<256xi32>) -> ()

  %c13 = AIE.core(%t73) { 
    
    %lb = arith.constant 0 : index
    %ub = arith.constant 1: index // 256*1= (256-2)*1
    %step = arith.constant 1 : index
    AIE.useLock(%lock73_14, "Acquire", 0) // start the timer
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%obj_fifo_in : !AIE.objectFifo<memref<256xi32>>, 1) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview = AIE.objectFifo.acquire<Produce>(%obj_fifo_out : !AIE.objectFifo<memref<256xi32>>, 1) : !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out = AIE.objectFifo.subview.access %obj_out_subview[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
        
      func.call @stencil_1d_3point(%row0,%obj_out) : (memref<256xi32>,  memref<256xi32>) -> ()
      AIE.objectFifo.release<Consume>(%obj_fifo_in : !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%obj_fifo_out : !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.useLock(%lock73_14, "Release", 0) // stop the timer
    AIE.end
  } { link_with="stencil_1d_3point.o" }

}

