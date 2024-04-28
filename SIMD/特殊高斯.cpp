#include<iostream>
#include<Windows.h>
#include<xmmintrin.h>//SSE
#include<emmintrin.h>//SSE2
#include<pmmintrin.h>//SSE3
#include<tmmintrin.h>//SSSE3
#include<smmintrin.h>//SSE4.1
#include<nmmintrin.h>//SSSE4.2
#include<immintrin.h>//AVX、AVX2
#include<sstream>
#include<fstream>
#include<iostream>
using namespace std;

const int spe_num = 1187; //记录消元子是否存在、被消元行的首项的数组位置。
const int line_num = 14921; //列数。
const int b_num = 37960; //被消元行数。
unsigned int Son[line_num][spe_num + 1] = { 0 };
unsigned int Be[line_num][spe_num + 2] = { 0 };
unsigned int Ans[line_num][spe_num + 1] = { 0 };
__m128 bik, son, vres;
__m256 bik1, son1, vres1;
void ge_spe(unsigned int Son[][spe_num + 1], unsigned int Be[][spe_num + 2])
{
    for (int i = 0; i < b_num; i++)
    {
        while (Be[i][spe_num + 1] == 0)
        {
            if (Son[Be[i][spe_num]][spe_num] == 1)
            {
                for (int k = 0; k < spe_num; k++)
                {
                    Be[i][k] =Be[i][k] ^ Son[Be[i][spe_num]][k];
                }

                //升格
                int S_num = 0;
                for (int num = 0; num < spe_num; num++)
                {
                
                    if (Be[i][num] != 0)
                    {
                        unsigned int temp = Be[i][num];
                        while (temp != 0)
                        {
                            temp = temp >> 1;
                            S_num++;
                        }
                        S_num = S_num + (spe_num - 1 - num) * 32;
                        break;
                    }
                }
                S_num -= 1;
                if (S_num < 0)
                {
                    Be[i][spe_num + 1] = 2;
                }
                else
                {
                    Be[i][spe_num] = S_num;
                }
            }
            else
            {
                for (int k = 0; k < spe_num; k++)
                {
                    Son[Be[i][spe_num]][k] = Be[i][k];
                }
                Son[Be[i][spe_num]][spe_num] = 1;
                Be[i][spe_num + 1] = 1;
            }
        }
    }

}

void ge_spe_sse(unsigned int Son[][spe_num + 1], unsigned int Be[][spe_num + 2])
{
    for (int i = 0; i < b_num; i++)
    {
        while (Be[i][spe_num + 1] == 0)
        {
            if (Son[Be[i][spe_num]][spe_num] == 1)
            {
                int k = 0;
                for (; k+4 <= spe_num; k+=4)
                {
                    bik = _mm_loadu_ps((float*) & Be[i][k]);
                    son = _mm_loadu_ps((float*)&Son[Be[i][spe_num]][k]);
                    vres = _mm_xor_ps(bik, son);
                    _mm_storeu_ps((float*)&Be[i][k], vres);
                }
                for (; k < spe_num; k++)
                {
                    Be[i][k] = Be[i][k] ^ Son[Be[i][spe_num]][k];
                }

                //升格
                int S_num = 0;
                for (int num = 0; num < spe_num; num++)
                {

                    if (Be[i][num] != 0)
                    {
                        unsigned int temp = Be[i][num];
                        while (temp != 0)
                        {
                            temp = temp >> 1;
                            S_num++;
                        }
                        S_num = S_num + (spe_num - 1 - num) * 32;
                        break;
                    }
                }
                S_num -= 1;
                if (S_num < 0)
                {
                    Be[i][spe_num + 1] = 2;
                }
                else
                {
                    Be[i][spe_num] = S_num;
                }
            }
            else
            {   
                int k = 0;
                for (; k+4 <= spe_num; k+=4)
                {
                    bik = _mm_loadu_ps((float*)&Be[i][k]);
                    _mm_storeu_ps((float*)&Son[Be[i][spe_num]][k], bik);
                }
                for (; k < spe_num; k++)
                {
                    Son[Be[i][spe_num]][k] = Be[i][k];
                }
                Son[Be[i][spe_num]][spe_num] = 1;
                Be[i][spe_num + 1] = 1;
            }
        }
    }

}

void ge_spe_avx(unsigned int Son[][spe_num + 1], unsigned int Be[][spe_num + 2])
{
    for (int i = 0; i < b_num; i++)
    {
        while (Be[i][spe_num + 1] == 0)
        {
            if (Son[Be[i][spe_num]][spe_num] == 1)
            {
                int k = 0;
                for (; k + 8 <= spe_num; k += 8)
                {
                    bik1 = _mm256_loadu_ps((float*)&Be[i][k]);
                    son1 = _mm256_loadu_ps((float*)&Son[Be[i][spe_num]][k]);
                    vres1 = _mm256_xor_ps(bik1, son1);
                    _mm256_storeu_ps((float*)&Be[i][k], vres1);
                }
                for (; k < spe_num; k++)
                {
                    Be[i][k] = Be[i][k] ^ Son[Be[i][spe_num]][k];
                }

                //升格
                int S_num = 0;
                for (int num = 0; num < spe_num; num++)
                {

                    if (Be[i][num] != 0)
                    {
                        unsigned int temp = Be[i][num];
                        while (temp != 0)
                        {
                            temp = temp >> 1;
                            S_num++;
                        }
                        S_num = S_num + (spe_num - 1 - num) * 32;
                        break;
                    }
                }
                S_num -= 1;
                if (S_num < 0)
                {
                    Be[i][spe_num + 1] = 2;
                }
                else
                {
                    Be[i][spe_num] = S_num;
                }
            }
            else
            {
                int k = 0;
                for (; k + 8 <= spe_num; k += 8)
                {
                    bik1 = _mm256_loadu_ps((float*)&Be[i][k]);
                    _mm256_storeu_ps((float*)&Son[Be[i][spe_num]][k], bik1);
                }
                for (; k < spe_num; k++)
                {
                    Son[Be[i][spe_num]][k] = Be[i][k];
                }
                Son[Be[i][spe_num]][spe_num] = 1;
                Be[i][spe_num + 1] = 1;
            }
        }
    }

}
int main()
{

    //数据读取。
    unsigned int a;
    ifstream infile("4/消元子.txt");
    char fin[10000] = { 0 };
    int index;
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int flag = 0;
        while (line >> a)
        {
            if (flag == 0)
            {
                index = a;
                flag = 1;
            }
            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Son[index][spe_num - 1 - j] += temp;
            Son[index][spe_num] = 1;
        }
    }

    ifstream infile1("4/被消元行.txt");
    char fin1[10000] = { 0 };
    index = 0;
    while (infile1.getline(fin1, sizeof(fin1)))
    {
        std::stringstream line(fin1);
        int flag = 0;

        while (line >> a)
        {
            if (flag == 0)
            {
                Be[index][spe_num] = a;
                flag = 1;
            }

            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Be[index][spe_num - 1 - j] += temp;
        }
        index++;
    }

    LARGE_INTEGER t1, t2, tc;
    QueryPerformanceFrequency(&tc);
    QueryPerformanceCounter(&t1);

    //ge_spe(Son, Be);
    //ge_spe_sse(Son,Be);
    ge_spe_avx(Son, Be);


    QueryPerformanceCounter(&t2);
    cout << ((t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart) << endl;

    /*for (int i = 0; i < 11; i++)
    {
        for (int j = 1175; j < spe_num+1; j++)
        {
            unsigned int x = Be[i][j];
            cout << x << " ";
        }
        cout << endl;
    }*/


    return 0;
}