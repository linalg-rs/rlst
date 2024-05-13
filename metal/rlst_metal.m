
#include "rlst_metal.h"

rlst_mtl_autorelease_pool_p rlst_mtl_new_autorelease_pool() {
  return (rlst_mtl_autorelease_pool_p)[[NSAutoreleasePool alloc] init];
}

void rlst_mtl_autorelease_pool_show_pools() {
  [NSAutoreleasePool showPools];
}

void rlst_mtl_autorelease_pool_drain(rlst_mtl_autorelease_pool_p p_pool) {
  [(NSAutoreleasePool *)p_pool drain];
}

rlst_mtl_device_p rlst_mtl_new_default_device() {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  return (rlst_mtl_device_p)device;
}

void rlst_mtl_device_release(rlst_mtl_device_p p_device) {
  id<MTLDevice> device = (id<MTLDevice>)p_device;
  [device release];
}

char *rlst_mtl_device_name(rlst_mtl_device_p p_device) {
  id<MTLDevice> device = (id<MTLDevice>)p_device;
  return (char *)[device.name UTF8String];
};

rlst_mtl_buffer_p rlst_mtl_device_new_buffer(rlst_mtl_device_p p_device,
                                             unsigned long length,
                                             unsigned int options) {
  id<MTLDevice> device = (id<MTLDevice>)p_device;
  return (rlst_mtl_buffer_p)[device newBufferWithLength:length options:options];
}

rlst_mtl_command_queue_p
rlst_mtl_device_new_command_queue(rlst_mtl_device_p p_device) {
  return (rlst_mtl_command_queue_p)[(id<MTLDevice>)p_device newCommandQueue];
}

void rlst_mtl_buffer_release(rlst_mtl_buffer_p p_buffer) {

  id<MTLBuffer> buffer = (id<MTLBuffer>)p_buffer;
  return [buffer release];
}

void *rlst_mtl_buffer_contents(rlst_mtl_buffer_p p_buffer) {
  id<MTLBuffer> buffer = (id<MTLBuffer>)p_buffer;
  return [buffer contents];
}

/* Command Queue */

void rlst_mtl_command_queue_release(rlst_mtl_command_queue_p p_queue) {
  [(id<MTLCommandQueue>)p_queue release];
}

rlst_mtl_command_buffer_p
rlst_mtl_command_queue_command_buffer(rlst_mtl_command_queue_p p_queue) {
  return (
      rlst_mtl_command_buffer_p)[(id<MTLCommandQueue>)p_queue commandBuffer];
}

/* Command Buffer */

void rlst_mtl_command_buffer_wait_until_completed(
    rlst_mtl_command_buffer_p p_command_buffer) {
  [(id<MTLCommandBuffer>)p_command_buffer waitUntilCompleted];
}

void rlst_mtl_command_buffer_commit(rlst_mtl_command_buffer_p p_command_buffer) {
  [(id<MTLCommandBuffer>)p_command_buffer commit];
}


rlst_mtl_compute_command_encoder_p
rlst_mtl_command_buffer_compute_command_encoder(
    rlst_mtl_command_buffer_p p_command_buffer, unsigned int dispatch_type) {
  return (rlst_mtl_compute_command_encoder_p)[(id<MTLCommandBuffer>)p_command_buffer
      computeCommandEncoderWithDispatchType:dispatch_type];
}

/* Matrix descriptor */

rlst_mtl_mps_matrix_descriptor_p rlst_mtl_mps_matrix_descriptor(
    unsigned long rows, unsigned long columns, unsigned long matrices,
    unsigned long rowBytes, unsigned long matrixBytes, unsigned int dataType) {
  MPSMatrixDescriptor *desc =
      [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                            columns:columns
                                           matrices:matrices
                                           rowBytes:rowBytes
                                        matrixBytes:matrixBytes
                                           dataType:dataType];
  return (rlst_mtl_mps_matrix_descriptor_p)desc;
}


size_t rlst_mtl_mps_matrix_descriptor_row_bytes_from_columns(unsigned long columns,
                                           unsigned int dataType) {
  return [MPSMatrixDescriptor rowBytesFromColumns:columns dataType:dataType];
}


unsigned long rlst_mtl_mps_matrix_descriptor_rows(rlst_mtl_mps_matrix_descriptor_p p_desc) {
  return [(MPSMatrixDescriptor* )p_desc rows];
}
unsigned long rlst_mtl_mps_matrix_descriptor_columns(rlst_mtl_mps_matrix_descriptor_p p_desc) {
  return [(MPSMatrixDescriptor* )p_desc columns];
}
unsigned long rlst_mtl_mps_matrix_descriptor_matrices(rlst_mtl_mps_matrix_descriptor_p p_desc) {
  return [(MPSMatrixDescriptor* )p_desc matrices];
}

unsigned long rlst_mtl_mps_matrix_descriptor_row_bytes(rlst_mtl_mps_matrix_descriptor_p p_desc) {
  return [(MPSMatrixDescriptor* )p_desc rowBytes];
}

unsigned long rlst_mtl_mps_matrix_descriptor_matrix_bytes(rlst_mtl_mps_matrix_descriptor_p p_desc) {
  return [(MPSMatrixDescriptor* )p_desc matrixBytes];
}



rlst_mtl_mps_matrix_p
rlst_mtl_mps_matrix(rlst_mtl_buffer_p p_buffer, unsigned long offset,
                    rlst_mtl_mps_matrix_descriptor_p p_desc) {
  return (rlst_mtl_mps_matrix_p)
      [[MPSMatrix alloc] initWithBuffer:(id<MTLBuffer>)p_buffer
                             offset: offset
                             descriptor:(MPSMatrixDescriptor *)p_desc];
}


void rlst_mtl_mps_matrix_release(rlst_mtl_mps_matrix_p p_matrix) {
  [(MPSMatrix*)p_matrix release];
}

rlst_mtl_mps_matrix_multiplication_p rlst_mtl_mps_matrix_multiplication(
    rlst_mtl_device_p p_device, bool transposeLeft, bool transposeRight,
    unsigned long resultRows, unsigned long resultColumns,
    unsigned long interiorColumns, double alpha, double beta) {
  return (rlst_mtl_mps_matrix_multiplication_p)
      [[MPSMatrixMultiplication alloc] initWithDevice:(id<MTLDevice>)p_device
                                        transposeLeft:transposeLeft
                                       transposeRight:transposeRight
                                           resultRows:resultRows
                                        resultColumns:resultColumns
                                      interiorColumns:interiorColumns
                                                alpha:alpha
                                                 beta:beta];
}

void rlst_mtl_mps_matrix_multiplication_release(rlst_mtl_mps_matrix_multiplication_p p_matmul) {
  [(MPSMatrixMultiplication *)p_matmul release];
}

void rlst_mtl_mps_matrix_multiplication_encode_to_command_buffer(rlst_mtl_mps_matrix_multiplication_p p_matmul, rlst_mtl_command_buffer_p p_commandBuffer, rlst_mtl_mps_matrix_p p_leftMatrix, rlst_mtl_mps_matrix_p p_rightMatrix, rlst_mtl_mps_matrix_p p_resultMatrix) {
  [(MPSMatrixMultiplication *)p_matmul encodeToCommandBuffer: (id<MTLCommandBuffer>)p_commandBuffer leftMatrix:(MPSMatrix *)p_leftMatrix rightMatrix: (MPSMatrix *)p_rightMatrix resultMatrix: (MPSMatrix *)p_resultMatrix];
}
