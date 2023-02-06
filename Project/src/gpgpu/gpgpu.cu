#include <gpgpu.h>
#include <algorithm>
#include <iostream>
#include <random>

__device__ float2 operator-(float2 a, float2 b) {
	return make_float2(a.x - b.x, a.y - b.y);
};

void GetGPGPUInfo() {
	cudaDeviceProp cuda_propeties;
	cudaGetDeviceProperties(&cuda_propeties, 0);
	std::cout << "maxThreadsPerBlock: " << cuda_propeties.maxThreadsPerBlock << std::endl;
}

void Init(Fox** f_i, int nf_i, Rabbit** r_i, int nr_i) {
	cudaMalloc((void**) f_i, sizeof(Fox) * nf_i);
	cudaMalloc((void**) r_i, sizeof(Fox) * nr_i);
}

__global__ void kernel_uv(cudaSurfaceObject_t surface, int32_t width, int32_t height, float time) {
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	float u = (float)x / width;
	float v = (float)y / height;
	float4 color = make_float4(u, v, cos(time), 1.0f);
	surf2Dwrite(color, surface, x * sizeof(float4), y);
}


__device__ void kernel_draw_rabbit(cudaSurfaceObject_t surface, int x, int y, int32_t width, int32_t height, Rabbit* rabbits, int32_t nb_rabbits) {
	float4 RABBIT_COLOR = make_float4(1.f, 1.f, 1.f, 1.0f);
	float2 uv;
	uv.x = (float)x / width;
	uv.y = (float)y / height;

	for (int n = 0; n < nb_rabbits; ++n) {
		if (rabbits[n].is_alive) {
			if (hypotf(rabbits[n].u - uv.x, rabbits[n].v - uv.y) < rabbits[n].radius) {
				surf2Dwrite(RABBIT_COLOR, surface, x * sizeof(float4), y);
			}
		}
	}
}

__device__ void kernel_draw_fox(cudaSurfaceObject_t surface, int x, int y, int32_t width, int32_t height, Fox* foxs, int32_t nb_foxs) {

	
	float2 uv;
	uv.x = (float)x / width;
	uv.y = (float)y / height;

	for (int n = 0; n < nb_foxs; ++n) {
		if (foxs[n].is_alive) {
			if (hypotf(foxs[n].u - uv.x, foxs[n].v - uv.y) < foxs[n].radius) {
				float degrade = 1.f - foxs[n].starvation / 50;
				float4 FOX_COLOR = make_float4(degrade, 0.f, 0.f, 1.0f);
				surf2Dwrite(FOX_COLOR, surface, x * sizeof(float4), y);
			}
		}
	}
}

__global__  void kernel_draw_map(cudaSurfaceObject_t surface, int32_t width, int32_t height, Fox* foxs, int32_t nb_foxs, Rabbit* rabbits, int32_t nb_rabbits) {
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	float4 color = make_float4(0.6f, 0.9f, 0.05f, 1.0f);

	surf2Dwrite(color, surface, x * sizeof(float4), y);
	kernel_draw_fox(surface, x, y, width, height, foxs, nb_foxs);
	kernel_draw_rabbit(surface, x, y, width, height, rabbits, nb_rabbits);
}

__global__ void kernel_copy(cudaSurfaceObject_t surface_in, cudaSurfaceObject_t surface_out) {
	int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	float4 color = make_float4(1.f, 0.f, 1.f, 1.0f);
	surf2Dread(&color, surface_in, x * sizeof(float4), y);
	surf2Dwrite(color, surface_out, x * sizeof(float4), y);
}

__device__ float fracf(float x)
{
	return x - floorf(x);
}

__device__ float random(float x, float y) {
	float t = 12.9898f * x + 78.233f * y;
	return abs(fracf(t * sin(t)));
}

__global__ void moveFoxs(Fox* fox, int32_t nb_foxs, Rabbit* rabbits, int32_t nb_rabbits) {
	int index = threadIdx.x; // 0 - nbFoxes
	if (fox[index].is_alive) {
		float angle = random(fox[index].u, fox[index].v) * 2.f - 1.f;
		float modifier = 0.5f;
		float temp = cos(angle * modifier) * fox[index].direction_u + sin(angle * modifier) * fox[index].direction_v;
		fox[index].direction_v = -fox[index].direction_u * sin(angle * modifier) + cos(angle * modifier) * fox[index].direction_v;
		fox[index].direction_u = temp;

		for (int i = 0; i < nb_rabbits; ++i) {
			if (rabbits[i].is_alive) {
				if (hypotf(fox[index].u - rabbits[i].u, fox[index].v - rabbits[i].v) < fox[index].detection_radius) {
					fox[index].direction_u = (rabbits[i].u - fox[index].u);
					fox[index].direction_v = (rabbits[i].v - fox[index].v);
				}
				if (hypotf(fox[index].u - rabbits[i].u, fox[index].v - rabbits[i].v) < fox[index].radius) {
					fox[index].starvation_modifier += 0.005f;
					fox[index].radius += 0.0002f;
					fox[index].detection_radius += 0.0006f;
					fox[index].eatenPrey++;
					rabbits[i].is_alive = false;
					fox[index].starvation -= 10.f;
				}
				//break;
			}
		}

		float norm = sqrt(pow(fox[index].direction_u, 2) + pow(fox[index].direction_v, 2));
		fox[index].direction_u = fox[index].direction_u / norm;
		fox[index].direction_v = fox[index].direction_v / norm;

		float new_pos_x = fox[index].u + fox[index].direction_u * fox[index].speed;
		float new_pos_y = fox[index].v + fox[index].direction_v * fox[index].speed;

		if ((new_pos_x + fox[index].radius < 1) && (new_pos_x - fox[index].radius > 0))
			fox[index].u = new_pos_x;
		else
			fox[index].direction_u = -fox[index].direction_u;
		if ((new_pos_y + fox[index].radius < 1) && (new_pos_y - fox[index].radius > 0))
			fox[index].v = new_pos_y;
		else
			fox[index].direction_v = -fox[index].direction_v;

		fox[index].starvation += 0.015f + fox[index].starvation_modifier;
		if (fox[index].starvation >= fox[index].max_starvation) {
			fox[index].is_alive = false;
		}

		if (fox[index].eatenPrey >= fox[index].max_eatenPrey) {
			int i = 0;
			while (fox[i].is_alive) {
				++i;
			}

			if (i != nb_foxs) {
				fox[index].eatenPrey = 0;
				fox[i].u = fox[index].u;
				fox[i].v = fox[index].v;
				fox[i].is_alive = true;
				fox[i].starvation = 0.f;
				fox[i].eatenPrey = 0;
				fox[i].starvation_modifier = 0.f;
				fox[i].radius = 0.01f;
				fox[i].detection_radius = 1.f / 30.f;
			}
		}
	}
}

__global__ void moveRabbits(Rabbit* rabbit, int32_t nb_rabbits) {
	int index = threadIdx.x;
	if (rabbit[index].is_alive) {
		float angle = random(rabbit[index].u, rabbit[index].v) * 2.f - 1.f;
		float modifier = 0.3f;

		float temp = cos(angle * modifier) * rabbit[index].direction_u + sin(angle * modifier) * rabbit[index].direction_v;
		rabbit[index].direction_v = -rabbit[index].direction_u * sin(angle * modifier) + cos(angle * modifier) * rabbit[index].direction_v;
		rabbit[index].direction_u = temp;

		float new_pos_x = rabbit[index].u + rabbit[index].direction_u * rabbit[index].speed;
		float new_pos_y = rabbit[index].v + rabbit[index].direction_v * rabbit[index].speed;

		if ((new_pos_x + rabbit[index].radius < 1) && (new_pos_x - rabbit[index].radius > 0))
			rabbit[index].u = new_pos_x;
		else
			rabbit[index].direction_u = - rabbit[index].direction_u;
		if ((new_pos_y + rabbit[index].radius < 1) && (new_pos_y - rabbit[index].radius > 0))
			rabbit[index].v = new_pos_y;
		else
			rabbit[index].direction_v = - rabbit[index].direction_v;

		if (abs(angle) < 0.0008f) {
			int i = 0;
			while (rabbit[i].is_alive){
				++i;
			}

			if (i != nb_rabbits){
				rabbit[i].is_alive = true;
				rabbit[i].u = rabbit[index].u;
				rabbit[i].v = rabbit[index].v;
			}
		}
	}
}

void DrawUVs(cudaSurfaceObject_t surface, int32_t width, int32_t height, float time) {
	dim3 threads(32, 32);
	dim3 blocks(32, 32);
	kernel_uv << <blocks, threads >> > (surface, width, height, time);
}

void DrawFoxs(Fox* foxs, int32_t nb_foxs, Rabbit* rabbits, int32_t nb_rabbits) {
	moveFoxs << < 1, nb_foxs >> > (foxs, nb_foxs, rabbits, nb_rabbits);
}

void DrawRabbits(Rabbit* rabbits, int32_t nb_rabbits) {
	moveRabbits << < 1, nb_rabbits >> > (rabbits, nb_rabbits);
}

void DrawMap(cudaSurfaceObject_t surface, int32_t width, int32_t height, Fox* foxs, int32_t nb_foxs, Rabbit* rabbits, int32_t nb_rabbits) {
	dim3 threads(32, 32);
	dim3 blocks(32, 32);

	kernel_draw_map << <blocks, threads >> > (surface, width, height, foxs, nb_foxs, rabbits, nb_rabbits);
	/*
	for (int i = 0; i < width; ++i) {
		for (int j = 0; j < height; ++j) {
			
			//kernel_draw_fox << <32 * 3, 1024 >> > (surface, i, j, width, height, foxs, nb_foxs);
		}
	}*/
}

void CopyTo(cudaSurfaceObject_t surface_in, cudaSurfaceObject_t surface_out, int32_t width, int32_t height) {
	dim3 threads(32, 32);
	dim3 blocks(32, 32);
	kernel_copy << <blocks, threads >> > (surface_in, surface_out);
}