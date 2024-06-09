#include<iostream>
#include <stdio.h>
#include<cstring>
#include<typeinfo>
#include <stdlib.h>
#include<cmath>
#include<mpi.h>
#include<windows.h>
#include<omp.h>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
using namespace std;
const int N = 2000;
float A[N][N];
float a[N][N];
int NUM_THREADS = 4;
//A[N][N]���������

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

//��ͨ��˹�Ĵ����㷨
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

//��ͨ��˹��avx�㷨
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
		//���в���
#pragma omp single
		{
			float tmp = A[k][k];
			for (int j = k + 1; j < N; j++)
			{
				A[k][j] = A[k][j] / tmp;
			}
			A[k][k] = 1.0;
		}

		//���в���
#pragma omp for schedule(static)
		for (int i = k + 1; i < N; i++)
		{
			float tmp = A[i][k];
			for (int j = k + 1; j < N; j++)
				A[i][j] = A[i][j] - tmp * A[k][j];
			A[i][k] = 0;
		}
		//�뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
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

void ge_mpi()
{
	int num_proc;//������
	int rank;//ʶ����ý��̵�rank��ֵ��0~size-1
	LARGE_INTEGER t1, t2, tc;
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//0�Ž��̡������񻮷�
	if (rank == 0)
	{
		re();
		QueryPerformanceFrequency(&tc);
		QueryPerformanceCounter(&t1);
		//���񻮷�
		for (int i = 0; i < N; i++)
		{
			int flag = i % num_proc;
			if (flag == rank)
				continue;
			else
				MPI_Send(&A[i], N, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
		}

		//
		for (int k = 0; k < N; k++)
		{
			//��ǰ�����Լ����̵����񡪡����г���
			if (int(k % num_proc) == rank)
			{
				for (int j = k + 1; j < N; j++)
					A[k][j] = A[k][j] / A[k][k];
				A[k][k] = 1.0;
				//���������������̷�����Ϣ
				for (int p = 0; p < num_proc; p++)
					if (p != rank)
						MPI_Send(&A[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
			}
			//��ǰ�в����Լ����̵����񡪡�������Ϣ
			else
			{
				MPI_Recv(&A[k], N, MPI_FLOAT, int(k % num_proc), 2,
					MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			for (int i = k + 1; i < N; i++)
			{
				if (int(i % num_proc) == rank)
				{
					for (int j = k + 1; j < N; j++)
						A[i][j] = A[i][j] - A[i][k] * A[k][j];
					A[i][k] = 0.0;
				}
			}
		}
		//

		//������0�Ž����Լ��������������������̴���֮��Ľ��
		for (int i = 0; i < N; i++)
		{
			int flag = i % num_proc;
			if (flag == rank)
				continue;
			else
				MPI_Recv(&A[i], N, MPI_FLOAT, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		QueryPerformanceCounter(&t2);
		cout << ((t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart) << endl;
	}
	//��������
	else
	{
		//��0�Ž����Ƚ�������
		for (int i = rank; i < N; i += num_proc)
		{
			MPI_Recv(&A[i], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		//ִ������


		for (int k = 0; k < N; k++)
		{
			//��ǰ�����Լ����̵����񡪡����г���
			if (int(k % num_proc) == rank)
			{
				for (int j = k + 1; j < N; j++)
					A[k][j] = A[k][j] / A[k][k];
				A[k][k] = 1.0;
				//���������������̷�����Ϣ
				for (int p = 0; p < num_proc; p++)
					if (p != rank)
						MPI_Send(&A[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
			}
			//��ǰ�в����Լ����̵����񡪡�������Ϣ
			else
			{
				MPI_Recv(&A[k], N, MPI_FLOAT, int(k % num_proc), 2,
					MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			for (int i = k + 1; i < N; i++)
			{
				if (int(i % num_proc) == rank)
				{
					for (int j = k + 1; j < N; j++)
						A[i][j] = A[i][j] - A[i][k] * A[k][j];
					A[i][k] = 0.0;
				}
			}
		}


		//��0�Ž����������֮�󣬽�������ص�0�Ž���
		for (int i = rank; i < N; i += num_proc)
		{
			MPI_Send(&A[i], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
		}
	}
}

void mpi_avx(float A[][N], int rank, int num_proc)
{
	__m256 vt1, va1, vaik1, vakj1, vaij1, vx1;
	for (int k = 0; k < N; k++)
	{
		vt1 = _mm256_set_ps(A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k]);
		int j;
		//��ǰ�����Լ����̵����񡪡����г���
		if (int(k % num_proc) == rank)
		{
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
			//���������������̷�����Ϣ
			for (int p = 0; p < num_proc; p++)
				if (p != rank)
					MPI_Send(&A[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
		}
		//��ǰ�в����Լ����̵����񡪡�������Ϣ
		else
		{
			MPI_Recv(&A[k], N, MPI_FLOAT, int(k % num_proc), 2,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}


		for (int i = k + 1; i < N; i++)
		{
			vaik1 = _mm256_set_ps(A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k]);
			int j;
			if (int(i % num_proc) == rank)
			{
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
				A[i][k] = 0.0;
			}
		}
	}
}

void mpi_avx() //���avx
{
	int num_proc;//������
	int rank;//ʶ����ý��̵�rank��ֵ��0~size-1
	LARGE_INTEGER t1, t2, tc;
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//0�Ž��̡������񻮷�
	if (rank == 0)
	{
		re();
		QueryPerformanceFrequency(&tc);
		QueryPerformanceCounter(&t1);
		//���񻮷�
		for (int i = 0; i < N; i++)
		{
			int flag = i % num_proc;
			if (flag == rank)
				continue;
			else
				MPI_Send(&A[i], N, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
		}

		//
		mpi_avx(A, rank, num_proc);
		//

		//������0�Ž����Լ��������������������̴���֮��Ľ��
		for (int i = 0; i < N; i++)
		{
			int flag = i % num_proc;
			if (flag == rank)
				continue;
			else
				MPI_Recv(&A[i], N, MPI_FLOAT, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		QueryPerformanceCounter(&t2);
		cout << ((t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart) << endl;
	}
	//��������
	else
	{
		//��0�Ž����Ƚ�������
		for (int i = rank; i < N; i += num_proc)
		{
			MPI_Recv(&A[i], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		//ִ������


		mpi_avx(A, rank, num_proc);


		//��0�Ž����������֮�󣬽�������ص�0�Ž���
		for (int i = rank; i < N; i += num_proc)
		{
			MPI_Send(&A[i], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
		}
	}

}

int main()
{
	init();
	/*LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);*/
	MPI_Init(NULL, NULL);
	ge_mpi();
	re();
	mpi_avx();
	MPI_Finalize();
	//ge();
	//avx_static();
	/*QueryPerformanceCounter(&t2);
	cout << ((t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart) << endl;*/

	return 0;
}

