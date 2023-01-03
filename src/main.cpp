#include <iostream>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <thread>
#include <cmath>

std::vector<uint8_t> image_data(512 * 512 * 3, 0);

std::vector<uint8_t> image_new(1024 * 1024 * 3, 0);

struct Complex {
	float real, imag;

	friend Complex operator*(Complex& z1, Complex& z2) {
		Complex temp;
		temp.real = z1.real * z2.real - z1.imag * z2.imag;
		temp.imag = z1.real * z2.imag + z1.imag * z2.real;
		return temp;
	}

	friend Complex operator+(Complex& z1, Complex& z2) {
		Complex temp;
		temp.real = z1.real + z2.real;
		temp.imag = z1.imag + z2.imag;
		return temp;
	}

	friend float modulus(Complex& z1) {
		return sqrt(pow(z1.real,2) + pow(z1.imag,2));
	}
};

void kernel(uint8_t& R, uint8_t& G, uint8_t& B, int x, int y) {
	R = x;
	G = y;
	B = 0;
}

void FillImage(std::vector<uint8_t>& input) {
	for (int i = 0; i < 512; ++i) {
		for (int j = 0; j < 512; ++j) {
			
			/*input[i * 512 * 3 + j * 3] = j/2;
			input[i * 512 * 3 + j * 3 + 1] = i/2;
			input[i * 512 * 3 + j * 3 + 2] = 0;*/

			kernel(input[i * 512 * 3 + j * 3], input[i * 512 * 3 + j * 3 + 1], input[i * 512 * 3 + j * 3 + 2], j / 2, i / 2);
		}
	}
}

void dispatch(int startX, int endX, int startY, int endY) {
	for (int i = startX; i < endX; ++i) {
		for (int j = startY; j < endY; ++j) {
			kernel(image_data[i * 512 * 3 + j * 3], image_data[i * 512 * 3 + j * 3 + 1], image_data[i * 512 * 3 + j * 3 + 2], j / 2, i / 2);
		}
	}
}

void main() {

	std::thread t0(
		dispatch,
		0,
		64,
		0,
		512
	);

	std::thread t1(
		dispatch,
		64,
		128,
		0,
		512
	);

	std::thread t2(
		dispatch,
		128,
		192,
		0,
		512
	);

	std::thread t3(
		dispatch,
		192,
		256,
		0,
		512
	);

	std::thread t4(
		dispatch,
		256,
		320,
		0,
		512
	);

	std::thread t5(
		dispatch,
		320,
		384,
		0,
		512
	);

	std::thread t6(
		dispatch,
		384,
		448,
		0,
		512
	);

	std::thread t7(
		dispatch,
		448,
		512,
		0,
		512
	);

	t0.join();
	t1.join();
	t2.join();
	t3.join();
	t4.join();
	t5.join();
	t6.join();
	t7.join();

	

	Complex z{ 0.f, 0.f };

	for (float u = 0; u<1; u+=1.f/1024) {
		for (float v = 0; v < 1; v+=1.f/1024) {
			Complex c{ u, v};
			unsigned int cmp = 0u;
			const int iterationMax = 5;
			while (modulus(z) < 2.f && cmp <= iterationMax) {
				z = z * z + c;
				++cmp;
			}
			const float red = static_cast<float>(cmp) / static_cast<float>(iterationMax);
			if (cmp >= iterationMax) {
				image_new[u * 1024 * 1024 * 3 + v * 1024 * 3] = 0;
			}
			else {
				image_new[u * 1024 * 1024 * 3 + v * 1024 * 3] = 255;
			}
			
		}
	}

	stbi_write_png("Hello_World.png", 512, 512, 3, image_data.data(), 512*3);
	stbi_write_png("notreProjet.png", 1024, 1024, 3, image_new.data(), 1024 * 3);

	
}
