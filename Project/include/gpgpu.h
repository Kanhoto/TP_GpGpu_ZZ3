#pragma once

#include <vector>

struct Circle {
	float u;
	float v;
	float radius;
};

struct Rabbit {
	float u;
	float v;
	float radius = 0.01f;
	float direction_u;
	float direction_v;
	float speed = 0.0005f;
	bool is_alive = false;
	//...
};

struct Fox {
	float u;
	float v;
	float radius = 0.01f;
	float detection_radius = 1.f / 30.f;
	float direction_u;
	float direction_v;
	float speed = 0.001f;
	float starvation = 0;
	float max_starvation = 50;
	float starvation_modifier = 0.f;
	int eatenPrey = 0;
	int max_eatenPrey = 15;
	bool is_alive = false;
	//...
};

void GetGPGPUInfo();
void Init(Fox** f_i, int nf_i, Rabbit** r_i, int nr_i);
void DrawUVs(cudaSurfaceObject_t surface, int32_t width, int32_t height, float time);
void DrawMap(cudaSurfaceObject_t surface, int32_t width, int32_t height, Fox* foxs, int32_t nb_foxs, Rabbit* rabbits, int32_t nb_rabbits);
void DrawFoxs(Fox* foxs, int32_t nb_foxs, Rabbit* rabbits, int32_t nb_rabbits);
void DrawRabbits(Rabbit* rabbits, int32_t nb_rabbits);
void CopyTo(cudaSurfaceObject_t surface_in, cudaSurfaceObject_t surface_out, int32_t width, int32_t height);