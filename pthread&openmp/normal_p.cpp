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
using namespace std;
const int N = 1000;
float A[N][N];
int worker_count = 5;
//A[N][N]是输入矩阵
__m128 vt, va, vaik, vakj, vaij, vx;//sse
__m256 vt1, va1, vaik1, vakj1, vaij1, vx1;//avx

struct threadParam_t
{
	int k; //消去的轮次
	int t_id; // 线程 id
};

void* threadFunc(void* param)
{

	__m256 va, vt, vx, vaij, vaik, vakj;

	threadParam_t* p = (threadParam_t*)param;
	int k = p->k; //消去的轮次
	int t_id = p->t_id; //线程编号
	int i = k + t_id + 1; //获取自己的计算任务
	for (int m = k + 1 + t_id; m < N; m += worker_count)
	{
		vaik = _mm256_set_ps(A[m][k], A[m][k], A[m][k], A[m][k], A[m][k], A[m][k], A[m][k], A[m][k]);
		int j;
		for (j = k + 1; j + 8 <= N; j += 8)
		{
			vakj = _mm256_loadu_ps(&(A[k][j]));
			vaij = _mm256_loadu_ps(&(A[m][j]));
			vx = _mm256_mul_ps(vakj, vaik);
			vaij = _mm256_sub_ps(vaij, vx);

			_mm256_store_ps(&A[i][j], vaij);
		}
		for (; j < N; j++)
			A[m][j] = A[m][j] - A[m][k] * A[k][j];

		A[m][k] = 0;
	}


	pthread_exit(NULL);
	return NULL;
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


int main()
{
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
		{
			A[i][j] = 0;
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

	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);
	
	for (int k = 0; k < N; k++)
	{
		vt1 = _mm256_set_ps(A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k]);
		int j;
		for (j = k + 1; j + 8 <= N; j += 8)
		{
			va1 = _mm256_loadu_ps(&(A[k][j]));
			va1 = _mm256_div_ps(va1, vt1);
			_mm256_store_ps(&(A[k][j]), va1);
		}

		for (; j < N; j++)
		{
			A[k][j] = A[k][j] * 1.0 / A[k][k];

		}
		A[k][k] = 1.0;

		

		pthread_t* handles = new pthread_t[worker_count];
		threadParam_t* param = new threadParam_t[worker_count];

		
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);

	
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_join(handles[t_id], NULL);

	}

	QueryPerformanceCounter(&t2);
	cout << ((t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart) << endl;
	
	return 0;
}

