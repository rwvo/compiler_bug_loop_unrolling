#include <vector>
#include <hc.hpp>

using namespace hc;

void matmul_sq_gpu_tiled(const float* A, const float* B, float* C, std::size_t rows)
{
  const std::size_t cols = rows;
  constexpr std::size_t tile_size = 32;
  const std::size_t tile_iterations = rows / tile_size;
  extent<2> ext(cols, rows);
  

  array_view<const float, 2> dev_A(ext, A);
  array_view<const float, 2> dev_B(ext, B);
  array_view<float, 2> dev_C(ext, C);
  dev_C.discard_data();

  parallel_for_each(ext.tile(tile_size, tile_size), 
		    [=](tiled_index<2> idx_c) [[hc]] {
      index<2> idx_a(idx_c.global[0], idx_c.local[1]); 
      index<2> idx_b(idx_c.local[0], idx_c.global[1]);

      tile_static float ts_a[tile_size][tile_size];
      tile_static float ts_b[tile_size][tile_size];
   
      float dot_product{0};
      for(std::size_t tile_iter = 0; tile_iter != tile_iterations; ++tile_iter){
	ts_a[idx_c.local[0]][idx_c.local[1]] = dev_A[idx_a];
	ts_b[idx_c.local[0]][idx_c.local[1]] = dev_B[idx_b];
	idx_c.barrier.wait();

	for(std::size_t i = 0; i != tile_size; ++i){
	  dot_product += ts_a[idx_c.local[0]][i] * ts_b[i][idx_c.local[1]];
	}

	idx_a[1] += tile_size;
	idx_b[0] += tile_size;
	idx_c.barrier.wait();
      }

      dev_C[idx_c.global] = dot_product;
    }).wait();

    dev_C.synchronize();
} 


int main(){
  std::size_t size = 1024;

  std::vector<float> A(size * size);
  std::vector<float> B(size * size);
  std::vector<float> C(size * size); 

  matmul_sq_gpu_tiled(A.data(), B.data(), C.data(), size);

  return 0;
}
