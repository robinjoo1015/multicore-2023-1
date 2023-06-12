#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define SPHERES 20

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF 2e10f
#define DIM 2048

struct Sphere {
    float   r,b,g;
    float   radius;
    float   x,y,z;
    // Set hit method to be executed on device
    __device__ float hit( float ox, float oy, float *n ) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx*dx + dy*dy < radius*radius) {
            float dz = sqrtf( radius*radius - dx*dx - dy*dy );
            *n = dz / sqrtf( radius * radius );
            return dz + z;
        }
        return -INF;
    }
};

// Set kernel method to be executed on device
__global__ void kernel(Sphere* s, unsigned char* ptr)
{
    // Calculate x and y from built-in variables
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = x + y*DIM;
	float ox = (x - DIM/2);
	float oy = (y - DIM/2);

	//printf("x:%d, y:%d, ox:%f, oy:%f\n",x,y,ox,oy);

	float r=0, g=0, b=0;
	float   maxz = -INF;
	for(int i=0; i<SPHERES; i++) {
		float   n;
		float   t = s[i].hit( ox, oy, &n );
		if (t > maxz) {
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
			maxz = t;
		} 
	}

	ptr[offset*4 + 0] = (int)(r * 255);
	ptr[offset*4 + 1] = (int)(g * 255);
	ptr[offset*4 + 2] = (int)(b * 255);
	ptr[offset*4 + 3] = 255;
}

void ppm_write(unsigned char* bitmap, int xdim,int ydim, FILE* fp)
{
	int i,x,y;
	fprintf(fp,"P3\n");
	fprintf(fp,"%d %d\n",xdim, ydim);
	fprintf(fp,"255\n");
	for (y=0;y<ydim;y++) {
		for (x=0;x<xdim;x++) {
			i=x+y*xdim;
			fprintf(fp,"%d %d %d ",bitmap[4*i],bitmap[4*i+1],bitmap[4*i+2]);
		}
		fprintf(fp,"\n");
	}
}

int main(int argc, char* argv[])
{
    unsigned char *bitmap;
    unsigned char *device_bitmap; // Bitmap on device
    Sphere *temp_s;
    Sphere *device_temp_s; // Sphere on device
    clock_t start_time, end_time;

    srand(time(NULL));

    FILE *fp = fopen("result.ppm","w");

    temp_s = (Sphere *)malloc(sizeof(Sphere) * SPHERES);
    // Allocate sphere memory on device
    cudaMalloc((void **)&device_temp_s, sizeof(Sphere) * SPHERES);
    for (int i=0; i<SPHERES; i++) {
		temp_s[i].r = rnd( 1.0f );
		temp_s[i].g = rnd( 1.0f );
		temp_s[i].b = rnd( 1.0f );
		temp_s[i].x = rnd( 2000.0f ) - 1000;
		temp_s[i].y = rnd( 2000.0f ) - 1000;
		temp_s[i].z = rnd( 2000.0f ) - 1000;
		temp_s[i].radius = rnd( 200.0f ) + 40;
	}

    bitmap=(unsigned char *)malloc(sizeof(unsigned char) * DIM * DIM * 4);
    // Allocate bitmap memory on device
    cudaMalloc((void **)&device_bitmap, sizeof(unsigned char) * DIM * DIM * 4);

    // Copy generated spheres to device
    cudaMemcpy(device_temp_s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice);

    // Set grids and threads
    dim3 grids(DIM / 32, DIM / 32, 1);
    dim3 threads(32, 32, 1);

    start_time = clock();

    // Execute kernel function
    kernel<<<grids, threads>>>(device_temp_s, device_bitmap);
    // Wait for device
    cudaDeviceSynchronize();

    end_time = clock();

    // Copy calculated spheres from device
    cudaMemcpy(bitmap, device_bitmap, sizeof(unsigned char) * DIM * DIM * 4, cudaMemcpyDeviceToHost);

    ppm_write(bitmap, DIM, DIM, fp);

    // Free device memory
    cudaFree(device_temp_s);
    cudaFree(device_bitmap);

    free(temp_s);
    free(bitmap);
    fclose(fp);

    // Print execution time
	printf("CUDA ray tracing: %.3lf sec\n", (end_time-start_time)/(double)1000);
	printf("[result.ppm] was generated\n");

    return 0;
}