#include<iostream>
#include<Windows.h>
#include<pthread.h>
#include<xmmintrin.h>//SSE
#include<emmintrin.h>//SSE2
#include<pmmintrin.h>//SSE3
#include<tmmintrin.h>//SSSE3
#include<smmintrin.h>//SSE4.1
#include<nmmintrin.h>//SSSE4.2
#include<immintrin.h>//AVX、AVX2
#include<omp.h>
using namespace std;
const int N = 1000;
float A[N][N];
float a[N][N];
int NUM_THREADS = 4;
//A[N][N]是输入矩阵

void init()
{
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
		{
			A[i][j] = 0;
			a[i][j] = 0;
		}

		A[i][i] = 1.0;

		for (int j = i + 1; j < N; j++)
		{
			A[i][j] = 1.0 * rand();
		}
	}

	for (int k = 0; k < N; k++) {
		for (int i = k + 1; i < N; i++) {
			for (int j = 0; j < N; j++) {
				A[i][j] += A[k][j];
			}
		}
	}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
		{
			a[i][j] = A[i][j];
			
		}
	}
}

void re()
{
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
		{
			A[i][j] = a[i][j];

		}
	}
}

//普通高斯的串行算法
void ge()
{
	for (int k = 0; k < N; k++)
	{
		for (int j = k + 1; j < N; j++)
		{
			A[k][j] = A[k][j] / A[k][k];
		}
		A[k][k] = 1.0;

		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				A[i][j] -= A[i][k] * A[k][j];
			}
			A[i][k] = 0;
		}

	}
	return;
}

//普通高斯的avx算法
void ge_avx()
{
	__m256 vt1, va1, vaik1, vakj1, vaij1, vx1;
	for (int k = 0; k < N; k++)
	{
		vt1 = _mm256_set_ps(A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k]);
		int j;
		for (j = k + 1; j + 8 <= N; j += 8)
		{
			va1 = _mm256_loadu_ps(&A[k][j]);
			va1 = _mm256_div_ps(va1, vt1);
			_mm256_storeu_ps(&A[k][j], va1);
		}

		for (; j < N; j++)
		{
			A[k][j] = A[k][j] / A[k][k];
		}

		A[k][k] = 1.0;

		for (int i = k + 1; i < N; i++)
		{
			vaik1 = _mm256_set_ps(A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k]);
			int j;
			for (j = k + 1; j + 8 <= N; j += 8)
			{
				vakj1 = _mm256_loadu_ps(&A[k][j]);
				vaij1 = _mm256_loadu_ps(&A[i][j]);
				vx1 = _mm256_mul_ps(vakj1, vaik1);
				vaij1 = _mm256_sub_ps(vaij1, vx1);
				_mm256_storeu_ps(&A[i][j], vaij1);
			}

			for (; j < N; j++)
			{
				A[i][j] = A[i][j] - A[k][j] * A[i][k];
			}
			A[i][k] = 0;
		}

	}
	return;
}

void op_ge_static()
{
#pragma omp parallel num_threads(NUM_THREADS)
	for (int k = 0; k < N; k++)
	{
		//串行部分
		#pragma omp single
		{
			float tmp = A[k][k];
			for (int j = k + 1; j < N; j++)
			{
				A[k][j] = A[k][j] / tmp;
			}
			A[k][k] = 1.0;
		}

		//并行部分
		#pragma omp for schedule(static)
		for (int i = k + 1; i < N; i++)
		{
			float tmp = A[i][k];
			for (int j = k + 1; j < N; j++)
				A[i][j] = A[i][j] - tmp * A[k][j];
			A[i][k] = 0;
		}
		 //离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void op_ge_dynamic()
{
#pragma omp parallel num_threads(NUM_THREADS)

	for (int k = 0; k < N; k++)
	{
		//串行部分
#pragma omp single
		{
			float tmp = A[k][k];
			for (int j = k + 1; j < N; j++)
			{
				A[k][j] = A[k][j] / tmp;
			}
			A[k][k] = 1.0;
		}

		//并行部分
#pragma omp for schedule(dynamic)
		for (int i = k + 1; i < N; i++)
		{
			float tmp = A[i][k];
			for (int j = k + 1; j < N; j++)
				A[i][j] = A[i][j] - tmp * A[k][j];
			A[i][k] = 0;
		}
		 //离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void op_ge_guided()
{
#pragma omp parallel num_threads(NUM_THREADS)

	for (int k = 0; k < N; k++)
	{
		//串行部分
#pragma omp single
		{
			float tmp = A[k][k];
			for (int j = k + 1; j < N; j++)
			{
				A[k][j] = A[k][j] / tmp;
			}
			A[k][k] = 1.0;
		}

		//并行部分
#pragma omp for schedule(guided)
		for (int i = k + 1; i < N; i++)
		{
			float tmp = A[i][k];
			for (int j = k + 1; j < N; j++)
				A[i][j] = A[i][j] - tmp * A[k][j];
			A[i][k] = 0;
		}
		 //离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void avx_static()
{
	__m256 vt1, va1, vaik1, vakj1, vaij1, vx1;
#pragma omp parallel num_threads(NUM_THREADS), private(vt1, va1, vaik1, vakj1, vaij1, vx1)
	for (int k = 0; k < N; k++)
	{
#pragma omp single
		{
			vt1 = _mm256_set_ps(A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k]);
			int j;
			for (j = k + 1; j + 8 <= N; j += 8)
			{
				va1 = _mm256_loadu_ps(&A[k][j]);
				va1 = _mm256_div_ps(va1, vt1);
				_mm256_storeu_ps(&A[k][j], va1);
			}

			for (; j < N; j++)
			{
				A[k][j] = A[k][j] / A[k][k];
			}

			A[k][k] = 1.0;
		}
#pragma omp for schedule(static)
		for (int i = k + 1; i < N; i++)
		{
			vaik1 = _mm256_set_ps(A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k]);
			int j;
			for (j = k + 1; j + 8 <= N; j += 8)
			{
				vakj1 = _mm256_loadu_ps(&A[k][j]);
				vaij1 = _mm256_loadu_ps(&A[i][j]);
				vx1 = _mm256_mul_ps(vakj1, vaik1);
				vaij1 = _mm256_sub_ps(vaij1, vx1);
				_mm256_storeu_ps(&A[i][j], vaij1);
			}

			for (; j < N; j++)
			{
				A[i][j] = A[i][j] - A[k][j] * A[i][k];
			}
			A[i][k] = 0;
		}

	}
	return;
}

void avx_dynamic()
{
	__m256 vt1, va1, vaik1, vakj1, vaij1, vx1;
#pragma omp parallel num_threads(NUM_THREADS), private(vt1, va1, vaik1, vakj1, vaij1, vx1)
	for (int k = 0; k < N; k++)
	{
#pragma omp single
		{
			vt1 = _mm256_set_ps(A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k]);
			int j;
			for (j = k + 1; j + 8 <= N; j += 8)
			{
				va1 = _mm256_loadu_ps(&A[k][j]);
				va1 = _mm256_div_ps(va1, vt1);
				_mm256_storeu_ps(&A[k][j], va1);
			}

			for (; j < N; j++)
			{
				A[k][j] = A[k][j] / A[k][k];
			}

			A[k][k] = 1.0;
		}
#pragma omp for schedule(dynamic)
		for (int i = k + 1; i < N; i++)
		{
			vaik1 = _mm256_set_ps(A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k]);
			int j;
			for (j = k + 1; j + 8 <= N; j += 8)
			{
				vakj1 = _mm256_loadu_ps(&A[k][j]);
				vaij1 = _mm256_loadu_ps(&A[i][j]);
				vx1 = _mm256_mul_ps(vakj1, vaik1);
				vaij1 = _mm256_sub_ps(vaij1, vx1);
				_mm256_storeu_ps(&A[i][j], vaij1);
			}

			for (; j < N; j++)
			{
				A[i][j] = A[i][j] - A[k][j] * A[i][k];
			}
			A[i][k] = 0;
		}

	}
	return;
}

void avx_guided()
{
	__m256 vt1, va1, vaik1, vakj1, vaij1, vx1;
#pragma omp parallel num_threads(NUM_THREADS), private(vt1, va1, vaik1, vakj1, vaij1, vx1)
	for (int k = 0; k < N; k++)
	{
#pragma omp single
		{
			vt1 = _mm256_set_ps(A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k]);
			int j;
			for (j = k + 1; j + 8 <= N; j += 8)
			{
				va1 = _mm256_loadu_ps(&A[k][j]);
				va1 = _mm256_div_ps(va1, vt1);
				_mm256_storeu_ps(&A[k][j], va1);
			}

			for (; j < N; j++)
			{
				A[k][j] = A[k][j] / A[k][k];
			}

			A[k][k] = 1.0;
		}
#pragma omp for schedule(guided)
		for (int i = k + 1; i < N; i++)
		{
			vaik1 = _mm256_set_ps(A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k]);
			int j;
			for (j = k + 1; j + 8 <= N; j += 8)
			{
				vakj1 = _mm256_loadu_ps(&A[k][j]);
				vaij1 = _mm256_loadu_ps(&A[i][j]);
				vx1 = _mm256_mul_ps(vakj1, vaik1);
				vaij1 = _mm256_sub_ps(vaij1, vx1);
				_mm256_storeu_ps(&A[i][j], vaij1);
			}

			for (; j < N; j++)
			{
				A[i][j] = A[i][j] - A[k][j] * A[i][k];
			}
			A[i][k] = 0;
		}

	}
	return;
}

int main()
{
	init();
	
	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);

	ge();
	
	QueryPerformanceCounter(&t2);
	cout  << ((t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart) << endl;

	re();
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);

	op_ge_static();

	QueryPerformanceCounter(&t2);
	cout << "op_ge_static():" << ((t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart) << endl;

	re();
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);

	op_ge_dynamic();

	QueryPerformanceCounter(&t2);
	cout << "op_ge_dynamic():" << ((t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart) << endl;

	re();
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);

	op_ge_guided();

	QueryPerformanceCounter(&t2);
	cout << "op_ge_guided():" << ((t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart) << endl;

	re();
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);

	ge_avx();

	QueryPerformanceCounter(&t2);
	cout << "ge_avx():" << ((t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart) << endl;

	re();
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);

	avx_static();

	QueryPerformanceCounter(&t2);
	cout << "avx_static():" << ((t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart) << endl;

	re();
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);

	avx_dynamic();

	QueryPerformanceCounter(&t2);
	cout << "avx_dynamic():" << ((t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart) << endl;

	re();
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);

	avx_guided();

	QueryPerformanceCounter(&t2);
	cout <<"avx_guided():" << ((t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart) << endl;


	return 0;
}

