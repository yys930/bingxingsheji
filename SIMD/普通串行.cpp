//#include<iostream>
//#include<Windows.h>
//#include<xmmintrin.h>//SSE
//#include<emmintrin.h>//SSE2
//#include<pmmintrin.h>//SSE3
//#include<tmmintrin.h>//SSSE3
//#include<smmintrin.h>//SSE4.1
//#include<nmmintrin.h>//SSSE4.2
//#include<immintrin.h>//AVX、AVX2
//using namespace std;
//const int N = 1000;
//float A[N][N];
////A[N][N]是输入矩阵
//__m128 vt, va, vaik, vakj, vaij, vx;//sse
//__m256 vt1, va1, vaik1, vakj1, vaij1, vx1;//avx
//
////普通高斯的串行算法
//void ge()
//{
//	for (int k = 0; k < N; k++)
//	{
//		for (int j = k + 1; j < N; j++)
//		{
//			A[k][j] = A[k][j] / A[k][k];
//		}
//		A[k][k] = 1.0;
//
//		for (int i = k + 1; i < N; i++)
//		{
//			for (int j = k + 1; j < N; j++)
//			{
//				A[i][j] -= A[i][k] * A[k][j];
//			}
//			A[i][k] = 0;
//		}
//		
//	}
//	return;
//}
//
////普通高斯的循环展开
//void ge_unroll()
//{
//	for (int k = 0; k < N; k++)
//	{
//		int j;
//		for (j = k + 1; j+2 <= N; j+=2)
//		{
//			A[k][j] = A[k][j] / A[k][k];
//			A[k][j+1] = A[k][j+1] / A[k][k];
//		}
//		for (; j < N; j++)
//		{
//			A[k][j] = A[k][j] / A[k][k];
//		}
//		A[k][k] = 1.0;
//
//		for (int i = k + 1; i < N; i++)
//		{
//			for (j = k + 1; j+2 <= N; j+=2)
//			{
//				A[i][j] -= A[i][k] * A[k][j];
//				A[i][j+1] -= A[i][k] * A[k][j+1];
//			}
//			for (; j < N; j++)
//			{
//				A[i][j] -= A[i][k] * A[k][j];
//			}
//			A[i][k] = 0;
//		}
//
//	}
//	return;
//}
//
////普通高斯的SSE算法
//void ge_sse()
//{
//	for (int k = 0; k < N; k++)
//	{
//		vt = _mm_set_ps(A[k][k], A[k][k], A[k][k], A[k][k]);
//		int j;
//		for (j = k + 1; j+4 <= N; j+=4)
//		{
//			va = _mm_loadu_ps(&A[k][j]);
//			va = _mm_div_ps(va, vt);
//			_mm_storeu_ps(&A[k][j], va);
//		}
//
//		for (; j < N; j++)
//		{
//			A[k][j] = A[k][j] / A[k][k];
//		}
//
//		A[k][k] = 1.0;
//
//		for (int i = k + 1; i < N; i++)
//		{
//			vaik = _mm_set_ps(A[i][k], A[i][k], A[i][k], A[i][k]);
//			int j;
//			for (j = k + 1; j+4 <= N; j+=4)
//			{
//				vakj = _mm_loadu_ps(&A[k][j]);
//				vaij = _mm_loadu_ps(&A[i][j]);
//				vx = _mm_mul_ps(vakj, vaik);
//				vaij = _mm_sub_ps(vaij, vx);
//				_mm_storeu_ps(&A[i][j], vaij);
//			}
//
//			for (; j < N; j++)
//			{
//				A[i][j] = A[i][j] - A[k][j] * A[i][k];
//			}
//			A[i][k] = 0;
//		}
//
//	}
//	return;
//}
//
////普通高斯的只优化div的sse算法
//void ge_sse_div()
//{
//	for (int k = 0; k < N; k++)
//	{
//		vt = _mm_set_ps(A[k][k], A[k][k], A[k][k], A[k][k]);
//		int j;
//		for (j = k + 1; j + 4 <= N; j += 4)
//		{
//			va = _mm_loadu_ps(&A[k][j]);
//			va = _mm_div_ps(va, vt);
//			_mm_storeu_ps(&A[k][j], va);
//		}
//
//		for (; j < N; j++)
//		{
//			A[k][j] = A[k][j] / A[k][k];
//		}
//
//		A[k][k] = 1.0;
//
//		for (int i = k + 1; i < N; i++)
//		{
//			for (int j = k + 1; j < N; j++)
//			{
//				A[i][j] -= A[i][k] * A[k][j];
//			}
//			A[i][k] = 0;
//		}
//		
//
//	}
//	return;
//}
//
////普通高斯的只优化sub的sse算法
//void ge_sse_sub()
//{
//	for (int k = 0; k < N; k++)
//	{
//		
//		for (int j = k + 1; j < N; j++)
//		{
//			A[k][j] = A[k][j] / A[k][k];
//		}
//
//		A[k][k] = 1.0;
//
//		for (int i = k + 1; i < N; i++)
//		{
//			vaik = _mm_set_ps(A[i][k], A[i][k], A[i][k], A[i][k]);
//			int j;
//			for (j = k + 1; j + 4 <= N; j += 4)
//			{
//				vakj = _mm_loadu_ps(&A[k][j]);
//				vaij = _mm_loadu_ps(&A[i][j]);
//				vx = _mm_mul_ps(vakj, vaik);
//				vaij = _mm_sub_ps(vaij, vx);
//				_mm_storeu_ps(&A[i][j], vaij);
//			}
//
//			for (; j < N; j++)
//			{
//				A[i][j] = A[i][j] - A[k][j] * A[i][k];
//			}
//			A[i][k] = 0;
//		}
//
//	}
//	return;
//}
//
////sse的对齐
//void sse()
//{
//	for (int k = 0; k < N; k++)
//	{
//		vt = _mm_set_ps(A[k][k], A[k][k], A[k][k], A[k][k]);
//		int j = k + 1;
//		while ((k * N + j) % 4 != 0)
//		{
//			A[k][j] = A[k][j] / A[k][k];
//			j++;
//		}
//		for (; j + 4 <= N; j += 4)
//		{
//			va = _mm_load_ps(&A[k][j]);
//			va = _mm_div_ps(va, vt);
//			_mm_store_ps(&A[k][j], va);
//		}
//
//		for (; j < N; j++)
//		{
//			A[k][j] = A[k][j] / A[k][k];
//		}
//
//		A[k][k] = 1.0;
//
//		for (int i = k + 1; i < N; i++)
//		{
//			vaik = _mm_set_ps(A[i][k], A[i][k], A[i][k], A[i][k]);
//			int j = k + 1;
//			while ((i * N + j) % 4 != 0)
//			{
//				A[i][j] -= A[i][k] * A[k][j];
//				j++;
//			}
//			for (; j + 4 <= N; j += 4)
//			{
//				vakj = _mm_load_ps(&A[k][j]);
//				vaij = _mm_load_ps(&A[i][j]);
//				vx = _mm_mul_ps(vakj, vaik);
//				vaij = _mm_sub_ps(vaij, vx);
//				_mm_store_ps(&A[i][j], vaij);
//			}
//
//			for (; j < N; j++)
//			{
//				A[i][j] = A[i][j] - A[k][j] * A[i][k];
//			}
//			A[i][k] = 0;
//		}
//
//	}
//	return;
//}
//
////普通高斯的avx算法
//void ge_avx() 
//{
//	for (int k = 0; k < N; k++)
//	{
//		vt1 = _mm256_set_ps(A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k]);
//		int j;
//		for (j = k + 1; j + 8 <= N; j += 8)
//		{
//			va1 = _mm256_loadu_ps(&A[k][j]);
//			va1 = _mm256_div_ps(va1, vt1);
//			_mm256_storeu_ps(&A[k][j], va1);
//		}
//
//		for (; j < N; j++)
//		{
//			A[k][j] = A[k][j] / A[k][k];
//		}
//
//		A[k][k] = 1.0;
//
//		for (int i = k + 1; i < N; i++)
//		{
//			vaik1 = _mm256_set_ps(A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k]);
//			int j;
//			for (j = k + 1; j + 8 <= N; j += 8)
//			{
//				vakj1 = _mm256_loadu_ps(&A[k][j]);
//				vaij1 = _mm256_loadu_ps(&A[i][j]);
//				vx1 = _mm256_mul_ps(vakj1, vaik1);
//				vaij1 = _mm256_sub_ps(vaij1, vx1);
//				_mm256_storeu_ps(&A[i][j], vaij1);
//			}
//
//			for (; j < N; j++)
//			{
//				A[i][j] = A[i][j] - A[k][j] * A[i][k];
//			}
//			A[i][k] = 0;
//		}
//
//	}
//	return;
//}
//
////普通高斯的只优化div的avx
//void ge_avx_div()
//{
//	for (int k = 0; k < N; k++)
//	{
//		vt1 = _mm256_set_ps(A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k]);
//		int j;
//		for (j = k + 1; j + 8 <= N; j += 8)
//		{
//			va1 = _mm256_loadu_ps(&A[k][j]);
//			va1 = _mm256_div_ps(va1, vt1);
//			_mm256_storeu_ps(&A[k][j], va1);
//		}
//
//		for (; j < N; j++)
//		{
//			A[k][j] = A[k][j] / A[k][k];
//		}
//		A[k][k] = 1.0;
//
//		for (int i = k + 1; i < N; i++)
//		{
//			for (int j = k + 1; j < N; j++)
//			{
//				A[i][j] -= A[i][k] * A[k][j];
//			}
//			A[i][k] = 0;
//		}
//
//	}
//	return;
//}
//
////普通高斯的avx只优化sub算法
//void ge_avx_sub()
//{
//	for (int k = 0; k < N; k++)
//	{
//		for (int j = k + 1; j < N; j++)
//		{
//			A[k][j] = A[k][j] / A[k][k];
//		}
//		A[k][k] = 1.0;
//
//		for (int i = k + 1; i < N; i++)
//		{
//			vaik1 = _mm256_set_ps(A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k]);
//			int j;
//			for (j = k + 1; j + 8 <= N; j += 8)
//			{
//				vakj1 = _mm256_loadu_ps(&A[k][j]);
//				vaij1 = _mm256_loadu_ps(&A[i][j]);
//				vx1 = _mm256_mul_ps(vakj1, vaik1);
//				vaij1 = _mm256_sub_ps(vaij1, vx1);
//				_mm256_storeu_ps(&A[i][j], vaij1);
//			}
//
//			for (; j < N; j++)
//			{
//				A[i][j] = A[i][j] - A[k][j] * A[i][k];
//			}
//			A[i][k] = 0;
//		}
//
//	}
//	return;
//}
//
////avx的对齐
//void avx()
//{
//	for (int k = 0; k < N; k++)
//	{
//		vt1 = _mm256_set_ps(A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k], A[k][k]);
//		int j = k + 1;
//		while ((k * N + j) % 8 != 0)
//		{
//			A[k][j] = A[k][j] / A[k][k];
//			j++;
//		}
//		for (; j + 8 <= N; j += 8)
//		{
//			va1 = _mm256_load_ps(&A[k][j]);
//			va1 = _mm256_div_ps(va1, vt1);
//			_mm256_store_ps(&A[k][j], va1);
//		}
//		for (; j < N; j++)
//		{
//			A[k][j] = A[k][j] / A[k][k];
//		}
//
//		A[k][k] = 1.0;
//
//		for (int i = k + 1; i < N; i++)
//		{
//			vaik1 = _mm256_set_ps(A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k], A[i][k]);
//			int j = k + 1;
//			while ((i * N + j) % 4 != 0)
//			{
//				A[i][j] -= A[i][k] * A[k][j];
//				j++;
//			}
//			for (; j + 8 <= N; j += 8)
//			{
//				vakj1 = _mm256_load_ps(&A[k][j]);
//				vaij1 = _mm256_load_ps(&A[i][j]);
//				vx1 = _mm256_mul_ps(vakj1, vaik1);
//				vaij1 = _mm256_sub_ps(vaij1, vx1);
//				_mm256_store_ps(&A[i][j], vaij1);
//			}
//
//			for (; j < N; j++)
//			{
//				A[i][j] = A[i][j] - A[k][j] * A[i][k];
//			}
//			A[i][k] = 0;
//		}
//
//	}
//	return;
//}
//
//int main()
//{
//	for (int i = 0; i < N; i++) {
//		for (int j = 0; j < N; j++)
//		{
//			A[i][j] = 0;
//		}
//
//		A[i][i] = 1.0;
//
//		for (int j = i + 1; j < N; j++)
//		{
//			A[i][j] = 1.0 * rand();
//		}
//	}
//
//	for (int k = 0; k < N; k++) {
//		for (int i = k + 1; i < N; i++) {
//			for (int j = 0; j < N; j++) {
//				A[i][j] += A[k][j];
//			}
//		}
//	}
//
//	LARGE_INTEGER t1, t2, tc;
//	QueryPerformanceFrequency(&tc);
//	QueryPerformanceCounter(&t1);
//	//ge();
//	//ge_unroll();
//	//ge_sse();
//	//ge_sse_div();
//	//ge_sse_sub();
//	//sse();
//	ge_avx();
//	//ge_avx_div();
//	//ge_avx_sub();
//	avx();
//	QueryPerformanceCounter(&t2);
//	cout << ((t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart) << endl;
//	/*ge();
//	for (int i = 0; i < N; i++)
//	{
//		cout << i << endl;
//		for (int j = 0; j < N; j++) {
//			cout << A[i][j] << " ";
//		}
//		cout << endl;
//	}*/
//	return 0;
//}
