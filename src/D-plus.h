/* paml.h 
*/

#if (!defined PAML_H)
#define PAML_H


#include <stdbool.h>
#include <signal.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <search.h>
#include <limits.h>
#include <float.h>
#include <time.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_gamma.h>
//#include <nlopt.h>

#define DBGLOCUS -1
#define MAXPARAMETERS  15
#define NGTREE 42
#define LBOUND 1.0E-5
#define MAX_LEGENDRE_POINTS 128
#define LEGENDRE_ORDER 16

#define square(a) ((a)*(a))
#define FOR(i,n) for(i=0; i<n; i++)
#define FPN(file) fputc('\n', file)
#define F0 stdout
#define min2(a,b) ((a)<(b)?(a):(b))
#define max2(a,b) ((a)>(b)?(a):(b))
#define max3(a,b,c) max2(max2(a,b), c)
#define swap2(a,b,y) { y=a; a=b; b=y; }
#define Pi  3.1415926535897932384626433832795

#define beep putchar('\a')
#define spaceming2(n) ((n)*((n)*2+9+2)*sizeof(double))

void err_put (char * message);
void trim(char *str);
void starttimer(void);
char* printtime(char timestr[]);
void error2(char * message);

int zero (double x[], int n);
double sum (double x[], int n);
int fillxc (double x[], double c, int n);
int xtoy (double x[], double y[], int n);
int abyx (double a, double x[], int n);
int axtoy(double a, double x[], double y[], int n);
int axbytoz(double a, double x[], double b, double y[], double z[], int n);
int identity (double x[], int n);
double distance (double x[], double y[], int n);
double innerp(double x[], double y[], int n);
double norm(double x[], int n);

int gradientB (int n, double x[], double f0, double g[],
    double (*fun)(double x[],int n), double space[], int xmark[]);
double fun_LineSearch (double t, double (*fun)(double x[],int n),
       double x0[], double p[], double x[], int n);
int H_end (double x0[], double x1[], double f0, double f1, double e1, double e2, int n);
double LineSearch2 (double(*fun)(double x[],int n), double *f, double x0[], 
    double p[], double h, double limit, double e, double space[], int n);
int ming2 (FILE *fout, double *f, double (*fun)(double x[], int n),
    int (*dfun)(double x[], double *f, double dx[], int n),
    double x[], double xb[][2], double space[], double e, int n);


#define PAML_RELEASE 1
#endif
