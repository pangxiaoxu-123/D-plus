/* D-plus - Robust introgression detection accounting for rate heterogeneity
 * Copyright (C) 2026 Your Name
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
/* 
 * Compile: cc -o D-plus -O3 -std=c99  -I ~/gsl-2.8/include -L ~/gsl-2.8/lib D-plus.c -fopenmp -lm -lgsl -lgslcblas
 */
#include "D-plus.h"

/* ================================================================
 * Section 1: Global data
 * ================================================================ */
double legendreNodes[MAX_LEGENDRE_POINTS];
double legendreWeights[MAX_LEGENDRE_POINTS];
static double y_node[MAX_LEGENDRE_POINTS];        /* (1+node)/2     */
static double weight_base[MAX_LEGENDRE_POINTS*MAX_LEGENDRE_POINTS]; /* w_i*w_j */
int MODEL_TO_GTREE_MAP[8][NGTREE];
int MODEL_TO_NG_MAP[8];
typedef struct {
	char jobname[256];
	char seqfile[256];
	int ndata;
	char s1[256];
	char s2[256];
	char o[256];
	int ABBA;
	int count[3];
	int event[3];
	int model;
	char *paramnames[MAXPARAMETERS];
	int np;
	int ngtree;
	int run;
	int RV;
	int repeat;
	int nthreads;
	int npoint;
	double *ILS_ref;
	double *Gprob[NGTREE];
	double *Sprob[NGTREE];
	double Glimi[NGTREE][2];
} CtlConfig;
typedef struct {
	int *Nij;
	double *lnLmax;
} DATA;

static const int DNA_TABLE[256] = {
	    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, // 0-15
	    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, // 16-31
	    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, // 32-47 
	    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, // 48-57 
	    15,  0, 11,  2, 12, 15, 15,  1, 13, 14, 15, 10,  5, 15, 15,  6, // 65-79 (A-O) -> A(65)=0, C(67)=2, G(71)=1...
	     8, 15, 15,  7,  3, 15,  9, 15, 15, 15, 15, 15, 15, 15, 15, 15, // 80-95 (P-Z) -> R(82)=6, S(83)=8, T(84)=3...
	    15,  0, 11,  2, 12, 15, 15,  1, 13, 14, 15, 10,  5, 15, 15,  6, // 97-111 (a-o)
	     8, 15, 15,  7,  3, 15,  9, 15, 15, 15, 15, 15, 15, 15, 15, 15, // 112-127 (p-z)
	    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, // 128+
	    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
	    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
	    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
	    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
	    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
	    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
	    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15
};

int parse_ctl_file(const char *filename, CtlConfig *config);
void parse_phylip_patterns();
void lnLMax (double space[]);
double random(); 
int RunModel (FILE *fout, double space[]);
void get_legendre_points_weights(int n, double legendreNodes[], double legendreWeights[]);
static double nlopt_obj_func(unsigned n, const double *x, double *grad, void *my_func_data);

int       m=16;
DATA      data;
double    para[MAXPARAMETERS]; 
CtlConfig config;
FILE      *fout;
FILE      *fbfgs;
char      timestr[96];
int       num_run=0;
double    Tau_int;

int    noisy = 0, Iround = 0, NFunCall = 0;
double SIZEp = 0;
int    AlwaysCenter = 0;


/* ================================================================
 * Section 2: main function
 * ================================================================ */
int main(int argc, char *argv[]) {
	srand((unsigned int)time(NULL)); 

	if (argc < 2) {
		fprintf(stderr, "Usage: %s <ctlfile>\n", argv[0]);
		return 1;
	}

	size_t space_size = spaceming2(MAXPARAMETERS);  
	double *space = (double*)malloc(space_size);  
	if(space == NULL) error2("\nError: oom space\n"); 

	if (!parse_ctl_file(argv[1], &config)) {
		fprintf(stderr, "Failed to parse ctl file\n");
		return 1;
	}
	
	char bfgsfile[256];
	strcpy(bfgsfile, config.jobname);
	strcat(bfgsfile, ".BFGS");
	char outfile[256];
	strcpy(outfile, config.jobname);
	strcat(outfile, ".out");
	if ((fbfgs = fopen(bfgsfile, "w")) == NULL) error2("failed to create bfgsfile..");
	if ((fout = fopen(outfile, "w")) == NULL) error2("failed to create outfile..");

	fprintf(stderr,"Parsed ctl file:\n");
	fprintf(stderr,"\toutfile  = %s.out\n", config.jobname);
	fprintf(stderr,"\tseqfile  = %s\n", config.seqfile);
	fprintf(stderr,"\tstree    = %s %s %s\n", config.s1, config.s2, config.o);
	fprintf(stderr,"\trepeat   = %d\n", config.repeat);
	fprintf(stderr,"\tevent    = %d %d %d\n", config.event[0],config.event[1],config.event[2]);
	if(config.run<2 && config.model>0) fprintf(stderr,"\tmodel   = M%d\n", config.model);
	fprintf(stderr,"\tnthreads = %d\n", config.nthreads);
	fprintf(fout,"Parsed ctl file:\n");
	fprintf(fout,"\toutfile  = %s.out\n", config.jobname);
	fprintf(fout,"\tseqfile  = %s\n", config.seqfile);
	fprintf(fout,"\tstree    = %s %s %s\n", config.s1, config.s2, config.o);
	fprintf(fout,"\trepeat   = %d\n", config.repeat);
	fprintf(fout,"\tevent    = %d %d %d\n", config.event[0],config.event[1],config.event[2]);
	if(config.run<2 &&  config.model>0) fprintf(fout,"\tmodel   = M%d\n", config.model);
	fprintf(fout,"\tnthreads = %d\n", config.nthreads);

	starttimer();   
	parse_phylip_patterns();
	lnLMax (space);
	/*
	for (int i = 0; i < 1000; i++) {
		printf("Locus %d: xxx=%d, yxx=%d, xyx=%d, xxy=%d, xyz=%d lnL=%8.6f\n",
		i, data.Nij[i*5+0], data.Nij[i*5+1], data.Nij[i*5+2], data.Nij[i*5+3], data.Nij[i*5+4], data.lnLmax[i]);
	}	
	exit(-1);
	*/

	get_legendre_points_weights(m,legendreNodes,legendreWeights);

	config.ILS_ref = (double*)malloc(m * m * sizeof(double));
	config.Gprob[0] = (double*)malloc((NGTREE*m*m)*sizeof(double));
	config.Sprob[0] = (double*)malloc((NGTREE*m*m*5)*sizeof(double));
	if(config.Gprob[0]==NULL || config.Sprob[0]==NULL || config.ILS_ref == NULL)
		error2("Memory allocation failed");
	for(int i=1; i<NGTREE; i++) {
		config.Gprob[i] = config.Gprob[i-1] + m*m;
		config.Sprob[i] = config.Sprob[i-1] + 5*m*m;
	}
	RunModel(fbfgs, space);
	fclose(fbfgs);
	fclose(fout);
	free(space);
	free(data.Nij);
	free(data.lnLmax);
	free(config.ILS_ref);
	free(config.Gprob[0]);
	free(config.Sprob[0]);

	return 0;
}

typedef struct {
	char e;       // 'B'/'U'/'G'
	int from;     //
	int to;       //
} Event;

/* ================================================================
 * Section 3: I/O
 * ================================================================ */
int parse_ctl_file(const char *filename, CtlConfig *config) {
	FILE *fp = fopen(filename, "r");
	if (!fp) {
		error2("Error opening ctl file");
	}
	config->model = -1;
	config->ABBA = 0;
	config->nthreads = 1;
	config->ndata = 10000;
	config->run = 2;
	config->RV = 1;
	config->repeat = 1;
	config->npoint = 16;
	char line[2048];
	while (fgets(line, sizeof(line), fp)) 
	{
		trim(line);  
		if (line[0] == '\0') continue; 
		if (line[0] == '*') continue; 
		char *eq = strchr(line, '=');
		if (!eq) continue;  
		*eq = '\0';
		char *key = line;
		char *val = eq + 1;
		trim(key);
		trim(val);
		if (key[0] == '\0' ||  val[0] == '\0') continue;

		if (strcasecmp(key, "jobname") == 0) {
			strncpy(config->jobname, val, sizeof(config->jobname) - 1);
			config->jobname[sizeof(config->jobname) - 1] = '\0';
		}else if (strcasecmp(key, "seqfile") == 0) {
			strncpy(config->seqfile, val, sizeof(config->seqfile) - 1);
			config->seqfile[sizeof(config->seqfile) - 1] = '\0';
		}else if (strcasecmp(key, "nloci") == 0) {
			config->ndata = atoi(val);
		}else if (strcasecmp(key, "nthreads") == 0) {
			config->nthreads = atoi(val);
		}else if (strcasecmp(key, "RV") == 0) {
			config->RV = atoi(val);
		}else if (strcasecmp(key, "ABBA") == 0) {
			config->ABBA = atoi(val);
		}else if (strcasecmp(key, "repeat") == 0) {
			config->repeat = atoi(val);
		}else if (strcasecmp(key, "npoints") == 0) {
			config->npoint = atoi(val);
			m = atoi(val);
		}else if (strcasecmp(key, "stree") == 0) {
			char buf[1024];
			strncpy(buf, val, sizeof(buf) - 1);
			buf[sizeof(buf) - 1] = '\0';
			
			char *tok = strtok(buf, " \t");
			if (tok) {
				strncpy(config->s1, tok, sizeof(config->s1) - 1);
				config->s1[sizeof(config->s1) - 1] = '\0';
			}else{
				error2("Invalid stree configuration parameter\n");	
			}
			
			tok = strtok(NULL, " \t");
			if (tok) {
				strncpy(config->s2, tok, sizeof(config->s2) - 1);
				config->s2[sizeof(config->s2) - 1] = '\0';
			}else{
				error2("Invalid stree configuration parameter\n");	
			}
			
			tok = strtok(NULL, " \t");
			if (tok) {
				strncpy(config->o, tok, sizeof(config->o) - 1);
				config->o[sizeof(config->o) - 1] = '\0';
			}else{
				error2("Invalid stree configuration parameter\n");	
			}
		}else if (strcasecmp(key, "run") == 0) {
			config->run = atoi(val);
			//printf ("### %d\n", config->run);
		}else if (strcasecmp(key, "model") == 0) {
			char buf0[1024];
			strncpy(buf0, val, sizeof(buf0) - 1);
			buf0[sizeof(buf0) - 1] = '\0';
			trim(buf0);
			char *event_str = strtok(buf0, "&");
			int valid_BU = 0;
			int valid_G = 0;
			
			while (event_str != NULL) {
				trim(event_str);
				//printf("Event: %s\n",event_str);
				if (strlen(event_str) == 0) {
					event_str = strtok(NULL, "&");
					continue;
				}
			
				Event event;
				char c1, c2, c3;
				int ret = sscanf(event_str, "%c %c %c", &c1, &c2, &c3);
				if (ret != 3){
					fprintf(stderr, "Error: format error in event '%s'\n", event_str);
					fclose(fp);
					exit(-1);
				}
				event.e = c1;
				event.from = c2 - '0';
				event.to = c3 - '0';
				if (event.from < 0 || event.to < 0) {
					fprintf(stderr, "Error: invalid node numbers in event '%s'\n", event_str);
					fclose(fp);
					exit(-1);
				}
				switch (event.e) {
					case 'G':
						if (event.from == 1 && event.to < 3) {
							config->event[0] = 1; //ghost-flow
							if (event.to == 2) valid_G = 1;
						} else {
							fprintf(stderr, "Error: G event '%s' must be with sister as the recipient\n", event_str);
							fclose(fp);
							exit(-1);
						}
						break;
					case 'B':
					case 'U':
						if (!((event.from == 2 && event.to == 3) || (event.from == 3 && event.to == 2) || (event.from == 1 && event.to == 3) || (event.from == 3 && event.to == 1))) {
							fprintf(stderr, "Error: B/U event '%s' must be between non-sister lineages\n", event_str);
							fclose(fp);
							exit(-1);
						} else {
							if (event.e == 'B') {
								config->event[1] = 1;
								config->event[2] = 1;
							} else {
								if (event.from == 3) config->event[1] = 1; // inflow
								else config->event[2] = 1; // outflow（from=2→to=3）
							}
							if (event.from == 1 || event.to == 1) valid_BU = 1;
						}
						break;

					default:
						break;
				}
				event_str = strtok(NULL, "&");
			}
			//000 → 0
			//001 → 1
			//010 → 2
			//011 → 3
			//100 → 4
			//101 → 5
			//110 → 6
			//111 → 7
			config->model = (config->event[0] << 2) | (config->event[1] << 1) | config->event[2];

			if (valid_BU == valid_G) {
				if (valid_G == 1 || valid_BU == 1){
					char temp[256];
					strncpy(temp, config->s1, sizeof(temp)-1);
					strncpy(config->s1, config->s2, sizeof(config->s1)-1);
					strncpy(config->s2, temp, sizeof(config->s2)-1);
				}
			}else{
				fprintf(stderr, "Conflicting events detected - cannot schedule both events simultaneously\n", event_str);
				fclose(fp);
				exit(-1);
			}
		}
	}
	fclose(fp);
	if (config->jobname[0] == '\0')
	{
		fprintf(stderr, "Error: jobname is not set in control file\n");
		exit(-1);
	}
	if (config->seqfile[0] == '\0')
	{
		fprintf(stderr, "Error: seqfile is not set in control file\n");
		exit(-1);
	}
	if (config->s1[0] == '\0' || config->s2[0] == '\0' || config->o[0] == '\0')
	{
		fprintf(stderr, "Error: stree is not set in control file\n");
		exit(-1);
	}
	if (config->run < 2 && (config->model < 0 || config->model > 7) )
	{
		fprintf(stderr, "Error: model is not set or contains an invalid value in control file\n");
		exit(-1);
	}
	return 1;
}


void parse_phylip_patterns() {
	FILE *fp = fopen(config.seqfile, "r");
	if (!fp) error2("Error opening phylip file");
	int loci = 0, all_loci = -1,  *n, i, pos, nseq, seqlen;
	double diff;
	int a, b, c, d;
	int site_if;
	int Invalid_locus[10000];
	memset(Invalid_locus, 0, config.ndata * sizeof(int));
	int m = 0;
	data.Nij = malloc(config.ndata * 5 * sizeof(int));
	memset(data.Nij, 0, config.ndata * 5 * sizeof(int));

	char name[256];
	char seq_buf[10001];
	char seqs[4][10001] = {0}; 

	while (1) 
	{
		int ret = fscanf(fp, "%d %d", &nseq, &seqlen);
		if (seqlen > 10000) seqlen = 10000;

		//header
		if (ret == EOF || ret != 2){
			break;
		}else if (ret == 2 &&  nseq < 3) {
			fprintf(stderr, "Warning: skip locus with sequence count < 3\n");
    			for(int k=0; k<nseq; k++){
        			fscanf(fp, "%255s %s");
    			}
				m++;
    			continue;	
		}
		all_loci++;

		// read sequences
		for (i = 0; i < nseq; i++) {
			if (fscanf(fp, "%*[^^]^%255s %10000s%*[^\n]",name, seq_buf) != 2) { 
				fclose(fp);
				free(data.Nij);
				data.Nij = NULL;
				error2("error in seq_name format\n");
			}
			if (strcmp(name, config.s1) == 0 && seqs[0][0] == '\0') {
				strcpy(seqs[0], seq_buf);
			}
			if (strcmp(name, config.s2) == 0 && seqs[1][0] == '\0') {
  				strcpy(seqs[1], seq_buf);
			}
			if (strcmp(name, config.o) == 0 && seqs[2][0] == '\0'){
		   		strcpy(seqs[2], seq_buf);
			}
			if (config.ABBA == 1 && strcmp(name, "Outgroup") == 0 && seqs[3][0] == '\0') {
				strcpy(seqs[3], seq_buf);
			}
		}
		
		if (strlen(seqs[0]) != seqlen || strlen(seqs[1]) != seqlen || strlen(seqs[2]) != seqlen || (config.ABBA == 1 && strlen(seqs[3]) != seqlen)) {
			if (strlen(seqs[0]) == 0){
				fprintf(stderr, "Warning: Species %s has no sequences in locus %d\n",config.s1 , all_loci + 1);
			}else if(strlen(seqs[1]) == 0){
				fprintf(stderr, "Warning: Species %s has no sequences in locus %d\n",config.s2 , all_loci + 1);
			}else if(strlen(seqs[2]) == 0){
				fprintf(stderr, "Warning: Species %s has no sequences in locus %d\n",config.o , all_loci + 1);
			}else if(config.ABBA == 1 && strlen(seqs[3]) == 0){
				fprintf(stderr, "Warning: Outgroup has no sequences in locus %d\n", all_loci + 1);
			}else{
				fprintf(stderr, "Warning: Sequence lengths differ in locus %d %d %d %d %d\n", all_loci + 1, strlen(seqs[0]), strlen(seqs[1]), strlen(seqs[2]), strlen(seqs[3]) );
			}
			Invalid_locus[m] = all_loci+1;			
			m++;
			continue;
		}

		// stat site patterns
		site_if = 0;
		n = data.Nij + loci * 5;
		for (pos = 0; pos < seqlen; pos++) {
			a = DNA_TABLE[(unsigned char)seqs[0][pos]];
			b = DNA_TABLE[(unsigned char)seqs[1][pos]];
			c = DNA_TABLE[(unsigned char)seqs[2][pos]];
			if (a >= 4 || b >= 4 || c >= 4 ) continue;
			if (config.ABBA == 1){
				d = DNA_TABLE[(unsigned char)seqs[3][pos]];
				if (d >= 4) continue;
			}
			//if (loci == DBGLOCUS && pos==0) fprintf(stderr, "%s, %s, %s\n", a, b, c);

			if (a == b && b == c && all_loci < config.ndata) {
				n[0]++; //xxx
			}else if (a != b && b == c) {
				if (all_loci < config.ndata) n[1]++; //yxx
				if (config.ABBA == 1 && a == d && site_if == 0) // ABBA 
				{
					config.count[0]++;
					site_if = 1;
				}
			}else if (b != a && a == c) {
				if (all_loci < config.ndata) n[2]++; //xyx
				if (config.ABBA == 1 && b == d && site_if == 0) // BABA
				{
					config.count[1]++;
					site_if = 1;
				}
			}else if (c != a && a == b) {
				if (all_loci < config.ndata) n[3]++; //xxy
				if (config.ABBA == 1 && c == d && site_if == 0) // BBAA 
				{
					config.count[2]++; 
					site_if = 1;
				}
			}else {
				n[4]++; //xyz
			}
			if (all_loci >= config.ndata && site_if) break;
		}
		memset(seqs, 0, sizeof(seqs));
		//printf ("%d: %d %d %d %d %d %d\n",all_loci,n[0],n[1],n[2],n[3],n[4]);
		if (n[0]+n[1]+n[2]+n[3]+n[4] == 0 && all_loci < config.ndata) 
		{
			Invalid_locus[m] = all_loci+1;			
			m++;
			continue; 
		}
		diff += (double)(n[1] + n[3])/(n[0]+n[1]+n[2]+n[3]+n[4]);
		if (all_loci < config.ndata) loci++;
	}
	config.ndata = loci;
	if (config.ndata == 0) error2("The number of valid loci is 0.\n");
	

	Tau_int = diff/loci/2; 
	//printf ("tau_int: %10.5f  %10.5f  %d\n", Tau_int, diff, seqlen);
	fprintf(stderr,"\tnloci    = %d\n", loci+m);
	fprintf(fout,"\tnloci    = %d\n", loci+m);
	if (Invalid_locus[0] != 0 )
	{
		printf ("Note: locus");
		for (int i = 0; i < m-1; i++) {
			printf (" %d,", Invalid_locus[i]);
		}
		printf (" %d contain no information\n", Invalid_locus[m-1]);
	}
	if (config.ABBA == 1){
		int count, temp;
		char sp[256];
			fprintf(stderr, "\nSite-Pattern Counts:\n");
		if (all_loci >= config.ndata){
			fprintf(stderr, "Note: All %d loci were used in the analysis.\n", all_loci+1);
			fprintf(fout,   "Note: All %d loci were used in the analysis.\n", all_loci+1);
		}
		if (config.count[0] < config.count[1]){
			count = config.count[0];
			config.count[0] = config.count[1];
			config.count[1] = count;
			strcpy(sp, config.s1);
			strcpy(config.s1, config.s2);
			strcpy(config.s2, sp);
			for (int i = 0; i < config.ndata; i++) {
				n = data.Nij + i * 5; 
    			temp = n[1];
			    n[1] = n[2];
				n[2] = temp;	
			}
			fprintf(stderr,"Note: Adjust P1 = %s and P2 = %s so that ABBA > BABA\n", config.s1, config.s2);
			fprintf(fout,"Note: Adjust P1 = %s and P2 = %s so that ABBA > BABA\n", config.s1, config.s2);
		}
		fprintf(stderr,"\tABBA = %d\n", config.count[0]);
		fprintf(stderr,"\tBABA = %d\n", config.count[1]);
		fprintf(stderr,"\tBBAA = %d\n", config.count[2]);
		fprintf(stderr,"\tDp   = %-10.4f\n", (double)(config.count[0]-config.count[1])/(double)(config.count[0]+config.count[1]+config.count[2]));
		fprintf(fout,"\tABBA = %d\n", config.count[0]);
		fprintf(fout,"\tBABA = %d\n", config.count[1]);
		fprintf(fout,"\tBBAA = %d\n", config.count[2]);
		fprintf(fout,"\tDp   = %-10.4f\n", (double)(config.count[0]-config.count[1])/(double)(config.count[0]+config.count[1]+config.count[2]));
	}
	fclose(fp);
}

/* ================================================================
 * Section 4: Probability computation
 * ================================================================ */
static int f_locus[5];

/* The (log) probability of site patterns given the unroot gene tree */
static inline void LogpFromb (double p[5], double b[3])
{
	const double c = -4.0/3.0;
	double e1 = exp(c * b[0]);
	double e2 = exp(c * b[1]);
	double e3 = exp(c * b[2]);
	double e1e2 = e1*e2, e1e3 = e1*e3, e2e3 = e2*e3;
	double e1e2e3 = e1e2*e3;

	p[0] = (1 + 3*e1e2 + 3*e1e3 + 3*e2e3 + 6*e1e2e3)/16; //xxx
	p[1] = (3 - 3*e1e2 - 3*e1e3 + 9*e2e3 - 6*e1e2e3)/16; //yxx
	p[2] = (3 - 3*e1e2 + 9*e1e3 - 3*e2e3 - 6*e1e2e3)/16; //xyx
	p[3] = (3 + 9*e1e2 - 3*e1e3 - 3*e2e3 - 6*e1e2e3)/16; //xxy
	p[4] = (6 - 6*e1e2 - 6*e1e3 - 6*e2e3 + 12*e1e2e3)/16; //xyz
	for(int i=0; i<5; i++) {
		if(p[i] < DBL_EPSILON) p[i] = DBL_EPSILON;
		p[i] = log(p[i]);
	}
}

// The (log) probability of data f_locus for the (unroot) gene tree with the branch lengths (obeying the multinomial distribution. The constant term is omitted.)
double lnLfromb (double b[3], int np)
{
	double lnL=0, p[5];
	int *f=f_locus;

	LogpFromb (p, b);
	lnL += (f[0] ? f[0] * p[0] : 0.0);
	lnL += (f[1] ? f[1] * p[1] : 0.0);
	lnL += (f[2] ? f[2] * p[2] : 0.0);
	lnL += (f[3] ? f[3] * p[3] : 0.0);
	lnL += (f[4] ? f[4] * p[4] : 0.0);
	return(-lnL);
}

// calculate max likelihood (lnL) by optimizing the branch lengths b for each locus
void lnLMax (double space[])
{
	double lnL, b[3], bb[3][2]={{1e-4,1},{1e-4,1},{1e-4,1}}, e=1e-7; //b: branch length; bb:the boundary for b	
	int nt, *f=f_locus, *n;
	data.lnLmax = malloc(config.ndata * sizeof(double));

	for(int locus=0; locus < config.ndata; locus++)  {
		n = data.Nij + locus * 5;
		// update f_locus
		for (int i=0; i<5; i++){f[i]=n[i];}
		nt = f[0]+f[1]+f[2]+f[3]+f[4];

		// initial value
		b[0]=(double)(f[1]+f[4])/nt;
		b[1]=(double)(f[2]+f[4])/nt;
		b[2]=(double)(f[3]+f[4])/nt;
		
		// calculate max likelihood (lnL) for each locus
		ming2(NULL, &lnL, lnLfromb, NULL, b, bb, space, e, 3);
		/*if (locus == DBGLOCUS) {
			printf("New Branches: %8.6f %8.6f %8.6f\n", b[0],b[1],b[2]);
		}*/

		data.lnLmax[locus] = -lnL-300; 
	}
}

// ghost inflow outflow
//para[MAXPARAMETERS]:
//Indix:        0        1       2        3       4        5        6         7        8        9     10          11       12         13        14 
//000-M0  theta_r theta_AB   tau_r   tau_AB  	  NA       NA        NA       NA       NA      NA     NA    lambda_A lambda_B  lambda_AB  lambda_C
//001-M1  theta_r theta_AB   tau_r   tau_AB       NA    tau_h   theta_B       NA       NA      NA  phi_o    lambda_A lambda_B  lambda_AB  lambda_C
//010-M2  theta_r theta_AB   tau_r   tau_AB       NA    tau_h        NA  theta_C       NA   phi_i     NA    lambda_A lambda_B  lambda_AB  lambda_C
//011-M3  theta_r theta_AB   tau_r   tau_AB       NA    tau_h   theta_B  theta_C       NA   phi_i  phi_o    lambda_A lambda_B  lambda_AB  lambda_C
//100-M4  theta_r theta_AB   tau_r   tau_AB    tau_g       NA        NA       NA    phi_g      NA     NA    lambda_A lambda_B  lambda_AB  lambda_C
//101-M5  theta_r theta_AB   tau_r   tau_AB    tau_g    tau_h   theta_B       NA    phi_g      NA  phi_o    lambda_A lambda_B  lambda_AB  lambda_C
//110-M6  theta_r theta_AB   tau_r   tau_AB    tau_g    tau_h        NA  theta_C    phi_g   phi_i     NA    lambda_A lambda_B  lambda_AB  lambda_C
//111-M7  theta_r theta_AB   tau_r   tau_AB    tau_g    tau_h   theta_B  theta_C    phi_g   phi_i  phi_o    lambda_A lambda_B  lambda_AB  lambda_C

// The parameters for each model
static int MODEL_to_PARA[8][MAXPARAMETERS] = {
    { 0, 1, 2, 3,-1,-1,-1,-1,-1,-1,-1, 4,-1,-1,-1},
    { 0, 1, 2, 3,-1, 4, 5,-1,-1,-1, 6, 7,-1,-1,-1},
    { 0, 1, 2, 3,-1, 4,-1, 5,-1, 6,-1, 7,-1,-1,-1},
    { 0, 1, 2, 3,-1, 4, 5, 6,-1, 7, 8, 9,-1,-1,-1},
    { 0, 1, 2, 3, 4,-1,-1,-1, 5,-1,-1, 6,-1,-1,-1},
    { 0, 1, 2, 3, 4, 5, 6,-1, 7,-1, 8, 9,-1,-1,-1},
    { 0, 1, 2, 3, 4, 5,-1, 6, 7, 8,-1, 9,-1,-1,-1},
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,-1,-1,-1}
};

// The possble gtree types for each model
const int GTREE_TO_PTREE_MAP[NGTREE] = { //the coalescent path for each gtree
    // gtree: 0-3 -> ptree: 0
    0, 0, 0, 0,                    // gtree 0-3; backbone path: 000
    // gtree: 4-7 -> ptree: 4  
    4, 4, 4, 4,                    // gtree 4-7; ghost path: 100
    // gtree: 8-11 -> ptree: 2
    2, 2, 2, 2,                    // gtree 8-11; inflow path: 010
    // gtree: 12-22 -> ptree: 1
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // gtree 12-22; outflow path: 001
    // gtree: 23-26 -> ptree: 3
    3, 3, 3, 3,                    // gtree 23-26; bidirectional path: 011
    // gtree: 27-31 -> ptree: 6
    6, 6, 6, 6, 6,                 // gtree 27-31; ghost-inflow path: 110
    // gtree: 32-37 -> ptree: 5
    5, 5, 5, 5, 5, 5,              // gtree 32-37; ghost-outflow path: 101
    // gtree: 38+ -> ptree: 7
    7, 7, 7, 7             	   // gtree 38-41; ghost-inflow-outflow path: 111
};
void Model_Gtree(){
	int pos, ptree;
	for (int model=0; model<8; model++)
	{
		pos=0;	
		// NGTREE = 42
		for (int gclass=0; gclass<NGTREE; gclass++)
		{
			MODEL_TO_GTREE_MAP[model][gclass] = -1;
	  		ptree = GTREE_TO_PTREE_MAP[gclass];
			// ptree is a subset of model (in the bitwise sense); e.g., 100(ptree)->100, 110, 101, 111(models)
	  		if((ptree & ~model) == 0){
				MODEL_TO_GTREE_MAP[model][pos] = gclass;
				pos++;
			}
			// Model 0: 0,1,2,3,-1,-1,...,-1
			// Model 1: 0,1,2,3,12,13,...,22,-1,-1,...,-1
			// ...
			// Model 7: 0,1,2,3,...,40,41
		}
		MODEL_TO_NG_MAP[model]=pos;
	}
}

// The correspondence between numbers and models
enum {M0, M1, M2, M3, M4, M5, M6, M7};
// one-to-one correspondence between numbers and events
enum {G,I,O};
static char *event_MAP[3] = {
	"Ghost",
	"Inflow",
	"Outflow"
}; 
// one-to-one correspondence between numbers and para_names 
enum {
	theta_r,       // 0
	theta_AB,      // 1
	tau_r,         // 2
	tau_AB,        // 3
	tau_g,         // 4
	tau_h,         // 5
	theta_B,       // 6
	theta_C,       // 7
	phi_g,         // 8
	phi_i,         // 9
	phi_o,         // 10
	lambda_A,      // 11
	lambda_B,	//12
	lambda_AB,     //13
	lambda_C       //14
};
static char *PARANAME_MAP[MAXPARAMETERS] = {
	"theta_R",
	"theta_P1P2",
	"tau_R",
	"tau_P1P2",
	"tau_G",
	"tau_H",
	"theta_P2",
	"theta_P3",
	"phi_g",
	"phi_i",
	"phi_o",
	"lambda_P1",
	"lambda_P1P2",
	"lambda_P2",
	"lambda_P3",
};
const int GTREE_IF_ILS_MAP[NGTREE] = {
	0, 1, 1, 1,
	0, 1, 1, 1,
	0, 1, 1, 1,
	0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	0, 1, 1, 1,
	0, 0, 1, 1, 1,
	0, 0, 0, 1, 1, 1,
	0, 1, 1, 1
};
const int GTREE_IF_ILS_MAP0[NGTREE] = {
	0, 1, 1, 1,
	0, 1, 1, 1,
	0, 1, 1, 1,
	0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
	0, 1, 1, 1,
	0, 0, 1, 1, 1,
	0, 0, 0, 1, 1, 1,
	0, 1, 1, 1
};
// Merging same gtrees though different coalescent paths (i.e., same gene-tree branch lengths for a given grid)
const int GTREE_IF_ILS_MAP1[NGTREE] = {
	0, 1, 0, 0,
	0, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
	0, 1, 0, 0,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0,
	0, 0, 0, 0
};
const int GTREE_IF_ILS_MAP2[NGTREE] = {
	0, 0, 1, 0,
	0, 0, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
	0, 0, 1, 0,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0,
	0, 0, 0, 0
};
const int GTREE_IF_ILS_MAP3[NGTREE] = {
	0, 0, 0, 1,
	0, 0, 0, 0,
	0, 0, 0, 1,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
	0, 0, 0, 1,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0,
	0, 0, 0, 0
};
const int GTREE_IF_ILS_MAP_G1[NGTREE] = {
	0, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0,
	0, 0, 1, 0, 0,
	0, 0, 0, 1, 0, 0,
	0, 1, 0, 0
};
const int GTREE_IF_ILS_MAP_G2[NGTREE] = {
	0, 0, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0,
	0, 0, 0, 1, 0,
	0, 0, 0, 0, 1, 0,
	0, 0, 1, 0
};
const int GTREE_IF_ILS_MAP_G3[NGTREE] = {
	0, 0, 0, 0,
	0, 0, 0, 1,
	0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0,
	0, 0, 0, 0, 1,
	0, 0, 0, 0, 0, 1,
	0, 0, 0, 1
};
const int GTREE_EQ_MAP_G[NGTREE] = {
	0, 0, 0, 0,
	1, 0, 0, 0,
	0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0,
	0, 1, 0, 0, 0,
	0, 0, 1, 0, 0, 0,
	1, 0, 0, 0
};
const int GTREE_DELETE[NGTREE] = {
	0, 0, 0, 0,
	0, 0, 0, 0,
	0, 1, 1, 1,
	0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
	0, 1, 1, 1,
	0, 1, 1, 1, 1,
	0, 0, 1, 1, 1, 1,
	1, 1, 1, 1
};

/* Get the upper limit value of the integral of each gtree with respect to coalescent times t1 and t2 (in coalescent unit)
*/
#define VAR_SUB(val) ((val)/(val+1.0))
void getGtreeIntegrationLimits() {
	double val, val1, val2;
	int ptree;
	for (int gclass = 0; gclass < NGTREE; gclass++) {
		config.Glimi[gclass][0] = 0.0;
		config.Glimi[gclass][1] = 0.0;
		
	  	ptree = GTREE_TO_PTREE_MAP[gclass];
		if ((ptree & ~config.model) != 0) continue;
		switch (gclass){
			//gclass == 0 000 
			case 0:
				val = 2*(para[tau_r]-para[tau_AB])/para[theta_AB];
				// Transform the variable and handle the infinite upper bound
				// Transform x/(x+1): [0,u] -> [0,u/(1+u)]; u/(1+u)=1 when u=infinite
				config.Glimi[gclass][0] = VAR_SUB(val);
				config.Glimi[gclass][1] = 1.0;
				break;
			//gclass == 4 100
			case 4:
				val = 2*(para[tau_g]-para[tau_r])/para[theta_r];
				config.Glimi[gclass][0] = VAR_SUB(val);
				config.Glimi[gclass][1] = 1.0;
				break;
			//gclass == 8 010
			case 8:
				val = 2*(para[tau_r]-para[tau_h])/para[theta_C];
				config.Glimi[gclass][0] = VAR_SUB(val);
				config.Glimi[gclass][1] = 1.0;
				break;
			//gclass == 12 001  2th coal_event in branch_AB
			case 12:
				val1 = 2*(para[tau_AB]-para[tau_h])/para[theta_B];
				val2 = 2*(para[tau_r]-para[tau_AB])/para[theta_AB];
				config.Glimi[gclass][0] = VAR_SUB(val1);
				config.Glimi[gclass][1] = VAR_SUB(val2);
				break;
			//gclass == 13 001 2th coal_event in branch_r
			case 13:
				val = 2*(para[tau_AB]-para[tau_h])/para[theta_B];
				config.Glimi[gclass][0] = VAR_SUB(val);
				config.Glimi[gclass][1] = 1.0;
				break;
			//gclass <= 16 001 ILS all coal_events in branch_AB
			case 14:
			case 15:
			case 16:
				val = 2*(para[tau_r]-para[tau_AB])/para[theta_AB];
				// let the frist parameter to be the ratio k=t1/t2 with interval [0,1] 
				config.Glimi[gclass][0] = 0.5;
				config.Glimi[gclass][1] = VAR_SUB(val);
				break;
			//gclass <= 19 001 ILS 1th coal_events in branch_AB and 2th coal_events in branch_r
			case 17:
			case 18:
			case 19:
				val = 2*(para[tau_r]-para[tau_AB])/para[theta_AB];
				config.Glimi[gclass][0] = VAR_SUB(val);
				config.Glimi[gclass][1] = 1.0;
				break;
			//gclass == 23 011
			case 23:
				val = 2*(para[tau_r]-para[tau_AB])/para[theta_AB];
				config.Glimi[gclass][0] = VAR_SUB(val);
				config.Glimi[gclass][1] = 1.0;
				break;
			//gclass == 27  110 & 1th coal-event in branch C
			case 27:
				val = 2*(para[tau_r]-para[tau_h])/para[theta_C];
				config.Glimi[gclass][0] = VAR_SUB(val);
				config.Glimi[gclass][1] = 1.0;
				break;
			//gclass == 28  110 & 1th coal-event in branch R
			case 28:
				val = 2*(para[tau_g]-para[tau_r])/para[theta_r];
				config.Glimi[gclass][0] = VAR_SUB(val);
				config.Glimi[gclass][1] = 1.0;
				break;
			//gclass == 32  101 & 1th coal-event in branch B
			case 32:
				val = 2*(para[tau_AB]-para[tau_h])/para[theta_B];
				config.Glimi[gclass][0] = VAR_SUB(val);
				config.Glimi[gclass][1] = 1.0;
				break;
			//gclass == 33  101 & 1th coal-event in branch AB
			case 33:
				val = 2*(para[tau_r]-para[tau_AB])/para[theta_AB];
				config.Glimi[gclass][0] = VAR_SUB(val);
				config.Glimi[gclass][1] = 1.0;
				break;
			//gclass == 34  101 & 1th coal-event in branch R
			//gclass == 38  111 
			case 34:
			case 38:
				val = 2*(para[tau_g]-para[tau_r])/para[theta_r];
				config.Glimi[gclass][0] = VAR_SUB(val);
				config.Glimi[gclass][1] = 1.0;
				break;
			//for other gclass: both coalescent events in the root branch with t1,t2 in [0,infinite]
			default:
		    		config.Glimi[gclass][0] = 1.0;
				config.Glimi[gclass][1] = 1.0;
				break;
			}
	}
}

/*
Given the model parameters, compute the following at each grid point (t1,t2):
	 1. Gprob: Probability of the gene tree under the specified model
	 2. Sprob: Probability of the five site patterns under the unrooted gene tree
Used for numerical integration in subsequent steps.

t1: The time interval for the first coalescent event, measured from the moment of entering the lineage to the completion of coalescence within that lineage.
t2: If the second coalescent event occurs in a different lineage than the first, t2 is measured from the moment of entering that lineage to the completion of coalescence within it;  
Otherwise (i.e., both events occur in the same lineage), t2 represents the time interval from the completion of the first coalescent event to the completion of the second coalescent event. 
In coalescent unit
except for gtree 14-16

Using ti (in coalescent unit) rather than xi (in mutation unit) improves the efficiency of numerical integration,
exp(-ti) vs 2/theta*exp(-xi*2/theta), the former decays much more slowly.
*/
void Cal_Gtree_info ()
{
	const double th_r = para[theta_r],   th_AB  = para[theta_AB];
	const double th_B = para[theta_B],   th_C   = para[theta_C];
	const double ta_r = para[tau_r],     ta_AB  = para[tau_AB];
	const double ta_g = para[tau_g],     ta_h   = para[tau_h];
	const double p_g  = para[phi_g],     p_i    = para[phi_i],    p_o = para[phi_o];
	const double k_A  = para[lambda_A],  k_B    = para[lambda_B];
	const double k_AB = para[lambda_AB], k_C    = para[lambda_C];

	const double C_000 = exp(- 2*(ta_r-ta_AB)/th_AB);
	const double C_010 = exp(- 2*(ta_r-ta_h)/th_C);
	const double C_001 = exp(- 2*(ta_AB-ta_h)/th_B - 6*(ta_r-ta_AB)/th_AB);
	const double C_011 = C_000;
	const double C_100 = exp(- 2*(ta_g-ta_r)/th_r);
	const double C_110 = C_100 * C_010;
	const double C_101 = exp(- 2*(ta_AB-ta_h)/th_B - 2*(ta_r-ta_AB)/th_AB - 2*(ta_g-ta_r)/th_r);
	const double C_111 = C_100;

	int gclass, ptree, grid, i0, i1;
	double upper1, upper2, t1_val, t2_val, y1_val, y2_val, denom, weight, C, curr_gprob, curr_gbran[3], p[5];
	for (gclass=0; gclass<NGTREE; gclass++)
	{
	  	ptree = GTREE_TO_PTREE_MAP[gclass];
		memset(config.Gprob[gclass], 0, m*m*sizeof(double));
		memset(config.Sprob[gclass], 0, 5*m*m*sizeof(double));
	  	if((ptree & ~config.model) != 0)
		{
			continue;
		}
		upper1 = config.Glimi[gclass][0];
		upper2 = config.Glimi[gclass][1];
		for (grid=0; grid<m*m; grid++)
		{
			i0=grid/m; i1=grid%m;
			// y->x: K(1+x)/2 x:[-1,1]-->y:[0,K]
			// t->y: y/(1-y) y:[0,K]-->t(branches in coalescent units, except for Gtree 14-16):[0,K/(1-K)]  ([0,T] with K=T/(T+1), where T is the upper limit of t)
			y1_val = upper1*y_node[i0]; //upper1*(1+legendreNodes[i0])*0.5; // legendreNodes[i0] is gauss point for the interval [-1,1]
			y2_val = upper2*y_node[i1];
			t1_val = y1_val/(1-y1_val);
			t2_val = y2_val/(1-y2_val);
			denom = (1-y1_val)*(1-y2_val);
			// Jacobi: (upper1*upper2*0.5*0.5) * 1.0/(denom*denom)
			weight = (upper1*upper2*0.5*0.5) * 1.0/(denom*denom)  * weight_base[i0*m+i1]; 
			switch (gclass){
			//parental trees:
			//#######################################000##############################backbone tree ((A,B):ta_AB, C):ta_r
				case 0: //000 ab|c
					curr_gprob = (1-p_g)*(1-p_i)*(1-p_o) * exp(-t1_val-t2_val);
					curr_gbran[0] = k_A*ta_AB + k_AB*(t1_val*th_AB/2);
					curr_gbran[1] = k_B*ta_AB + k_AB*(t1_val*th_AB/2);
					curr_gbran[2] = k_C*ta_r + 2*t2_val*th_r/2 + k_AB*(ta_r-ta_AB-t1_val*th_AB/2);
					break;
				case 1: //000 ILS ab|c
					//curr_gprob = (1-p_g)*(1-p_i)*(1-p_o) * exp(-3*t1_val-t2_val - 2*(ta_r-ta_AB)/th_AB);
					config.ILS_ref[grid] = exp(-3*t1_val-t2_val);
					C = (1-p_g)*(1-p_i)*(1-p_o) * C_000;
					curr_gbran[0] = k_A*ta_AB + k_AB*(ta_r-ta_AB) + t1_val*th_r/2;
					curr_gbran[1] = k_B*ta_AB + k_AB*(ta_r-ta_AB) + t1_val*th_r/2;
					curr_gbran[2] = k_C*ta_r + (2*t2_val+t1_val)*th_r/2;
					break;
				case 2: //000 ILS a|bc
					//curr_gprob = (1-p_g)*(1-p_i)*(1-p_o) * exp(-3*t1_val-t2_val -2*(ta_r-ta_AB)/th_AB);
					C = (1-p_g)*(1-p_i)*(1-p_o) * C_000;
					curr_gbran[0] = k_A*ta_AB + k_AB*(ta_r-ta_AB) + (2*t2_val+t1_val)*th_r/2;
					curr_gbran[1] = k_B*ta_AB + k_AB*(ta_r-ta_AB) + t1_val*th_r/2;
					curr_gbran[2] = k_C*ta_r + t1_val*th_r/2;
					break;
				case 3: //000 ILS b|ac
					//curr_gprob = (1-p_g)*(1-p_i)*(1-p_o) * exp(-3*t1_val-t2_val -2*(ta_r-ta_AB)/th_AB);
					C = (1-p_g)*(1-p_i)*(1-p_o) * C_000;
					curr_gbran[0] = k_A*ta_AB + k_AB*(ta_r-ta_AB) + t1_val*th_r/2;
					curr_gbran[1] = k_B*ta_AB + k_AB*(ta_r-ta_AB) + (2*t2_val+t1_val)*th_r/2;
					curr_gbran[2] = k_C*ta_r + t1_val*th_r/2;
					break;
			//#######################################100##############################
			//Here, assuming time of ghost introgression = 0;
				case 4: //100 a|bc       
					curr_gprob = p_g*(1-p_i)*(1-p_o)*exp(-t1_val-t2_val);
					curr_gbran[0] = ta_g + 2*t2_val*th_r/2 + (ta_g-ta_r-t1_val*th_r/2); // associated with ghost-flow time;
					curr_gbran[1] = k_B*ta_AB + k_AB*(ta_r-ta_AB)+ t1_val*th_r/2;
					curr_gbran[2] = k_C*ta_r + t1_val*th_r/2; 
					break;
				case 5: //100 ILS ab|c
					//curr_gprob = p_g*(1-p_i)*(1-p_o) * exp(-3*t1_val-t2_val - 2*(ta_g-ta_r)/th_r);
					C = p_g*(1-p_i)*(1-p_o) * C_100;
					curr_gbran[0] = ta_g + t1_val*th_r/2;
					curr_gbran[1] = k_B*ta_AB + k_AB*(ta_r-ta_AB)+ (ta_g-ta_r) + t1_val*th_r/2;
					curr_gbran[2] = k_C*ta_r + (ta_g-ta_r) + (2*t2_val+t1_val)*th_r/2;
					break;
				case 6: //100 ILS a|bc
					//curr_gprob = p_g*(1-p_i)*(1-p_o) * exp(-3*t1_val-t2_val - 2*(ta_g-ta_r)/th_r);
					C = p_g*(1-p_i)*(1-p_o) * C_100;
					curr_gbran[0] = ta_g + (2*t2_val+t1_val)*th_r/2;
					curr_gbran[1] = k_B*ta_AB + k_AB*(ta_r-ta_AB)+ (ta_g-ta_r) + t1_val*th_r/2;
					curr_gbran[2] = k_C*ta_r + (ta_g-ta_r) + t1_val*th_r/2;
					break;
				case 7: //100 ILS b|ac
					//curr_gprob = p_g*(1-p_i)*(1-p_o) * exp(-3*t1_val-t2_val - 2*(ta_g-ta_r)/th_r);
					C = p_g*(1-p_i)*(1-p_o) * C_100;
					curr_gbran[0] = ta_g + t1_val*th_r/2;
					curr_gbran[1] = k_B*ta_AB + k_AB*(ta_r-ta_AB)+ (ta_g-ta_r) + (2*t2_val+t1_val)*th_r/2;
					curr_gbran[2] = k_C*ta_r + (ta_g-ta_r) + t1_val*th_r/2;
					break;
			//#######################################010##############################inflow tree
				case 8: //010 a|bc
					curr_gprob = (1-p_g)*p_i*(1-p_o)*exp(-t1_val-t2_val);
					curr_gbran[0] = k_A*ta_AB + k_AB*(ta_r-ta_AB) + 2*t2_val*th_r/2 + k_C*(ta_r-ta_h-t1_val*th_C/2);
					curr_gbran[1] = k_B*ta_h + k_C*(t1_val*th_C/2);
					curr_gbran[2] = k_C*(ta_h + t1_val*th_C/2);
					break;
				case 9: //010 ILS ab|c
					C = (1-p_g)*p_i*(1-p_o) * C_010;
					//curr_gprob = (1-p_g)*p_i*(1-p_o) * exp(-2/th_C*(ta_r-ta_h) - 3*t1_val-t2_val); 
					curr_gbran[0] = k_A*ta_AB + k_AB*(ta_r-ta_AB) + t1_val*th_r/2;
					curr_gbran[1] = k_B*ta_h + k_C*(ta_r-ta_h) + t1_val*th_r/2;
					curr_gbran[2] = k_C*ta_r + (2*t2_val+t1_val)*th_r/2;
					break;
				case 10: //010 ILS a|bc
					//curr_gprob = (1-p_g)*p_i*(1-p_o) * exp(-2/th_C*(ta_r-ta_h) - 3*t1_val-t2_val); 
					C = (1-p_g)*p_i*(1-p_o) * C_010;
					curr_gbran[0] = k_A*ta_AB + k_AB*(ta_r-ta_AB) + (2*t2_val+t1_val)*th_r/2;
					curr_gbran[1] = k_B*ta_h + k_C*(ta_r-ta_h) + t1_val*th_r/2;
					curr_gbran[2] = k_C*ta_r + t1_val*th_r/2;
					break;
				case 11: //010 ILS b|ac
					//curr_gprob = (1-p_g)*p_i*(1-p_o) * exp(-2/th_C*(ta_r-ta_h) - 3*t1_val-t2_val); 
					C = (1-p_g)*p_i*(1-p_o) * C_010;
					curr_gbran[0] = k_A*ta_AB + k_AB*(ta_r-ta_AB) + t1_val*th_r/2;
					curr_gbran[1] = k_B*ta_h + k_C*(ta_r-ta_h) + (2*t2_val+t1_val)*th_r/2;
					curr_gbran[2] = k_C*ta_r + t1_val*th_r/2;
					break;
			//#######################################001##############################outflow tree ((B,C):ta_h,A):ta_AB
				case 12: //001 & a|bc & 2th coal_event in branch_AB
					curr_gprob = (1-p_g)*(1-p_i)*p_o * exp(-t1_val-t2_val);
					curr_gbran[0] = k_A*ta_AB + k_AB*2*t2_val*th_AB/2 + k_B*(ta_AB-ta_h-t1_val*th_B/2.0);
					curr_gbran[1] = k_B*(ta_h+t1_val*th_B/2);
					curr_gbran[2] = k_C*ta_h + k_B*t1_val*th_B/2;
					break;
				case 13: //001 & a|bc & 2th coal_event in branch_r
					curr_gprob = (1-p_g)*(1-p_i)*p_o * exp(-2*(ta_r-ta_AB)/th_AB -t1_val-t2_val);
					curr_gbran[0] = k_A*ta_AB + k_AB*(ta_r-ta_AB)*2 + 2*t2_val*th_r/2 + k_B*(ta_AB-ta_h-t1_val*th_B/2);
					curr_gbran[1] = k_B*(ta_h+t1_val*th_B/2);
					curr_gbran[2] = k_C*ta_h + k_B*t1_val*th_B/2;
					break;
				case 14: //001 ILS ab|c & all coal_events in branch_AB  
					 // t1==>t1/(t1+t2) 
					 // t2==>t1+t2
					curr_gprob = (1-p_g)*(1-p_i)*(p_o) * t2_val*exp(-2/th_B*(ta_AB-ta_h) - 2*t1_val*t2_val-t2_val);
					curr_gbran[0] = k_A*ta_AB + k_AB*t1_val*t2_val*th_AB/2;
					curr_gbran[1] = k_B*ta_AB + k_AB*t1_val*t2_val*th_AB/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h) + k_AB*(2*t2_val-t1_val*t2_val)*th_AB/2;
					break;
				case 15: //001 ILS a|bc & all coal_events in branch_AB
					//curr_gprob = (1-p_g)*(1-p_i)*(p_o) * exp(-2/th_B*(ta_AB-ta_h)) * t2_val*exp(-2*t1_val*t2_val-t2_val);
					//curr_gprob = (1-p_g)*(1-p_i)*(p_o) * t2_val*exp(-2/th_B*(ta_AB-ta_h) - 2*t1_val*t2_val-t2_val);
					curr_gbran[0] = k_A*ta_AB + k_AB*(2*t2_val-t1_val*t2_val)*th_AB/2;
					curr_gbran[1] = k_B*ta_AB + k_AB*t1_val*t2_val*th_AB/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h) + k_AB*t1_val*t2_val*th_AB/2;
					break;
				case 16: //001 ILS ac|b & all coal_events in branch_AB
					//curr_gprob = (1-p_g)*(1-p_i)*(p_o) * t2_val*exp(-2/th_B*(ta_AB-ta_h) - 2*t1_val*t2_val-t2_val);
					curr_gbran[0] = k_A*ta_AB + k_AB*t1_val*t2_val*th_AB/2;
					curr_gbran[1] = k_B*ta_AB + k_AB*(2*t2_val-t1_val*t2_val)*th_AB/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h) + k_AB*t1_val*t2_val*th_AB/2;
					break;
				case 17: //001 ILS ab|c 1th coal_event in branch_AB and 2th coal_event in branch_r
					curr_gprob = (1-p_g)*(1-p_i)*(p_o) * exp(-2/th_B*(ta_AB-ta_h) + (-2/th_AB*(ta_r-ta_AB)+t1_val)  + (-3*t1_val-t2_val));
					curr_gbran[0] = k_A*ta_AB + k_AB*t1_val*th_AB/2;
					curr_gbran[1] = k_B*ta_AB + k_AB*t1_val*th_AB/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h) + k_AB*(2*(ta_r-ta_AB)-t1_val*th_AB/2) + 2*t2_val*th_r/2;
					break;
				case 18: //001 ILS a|bc 1th coal_event in branch_AB and 2th coal_event in branch_r
					//curr_gprob = (1-p_g)*(1-p_i)*(p_o) * exp(-2/th_B*(ta_AB-ta_h) + (-2/th_AB*(ta_r-ta_AB)+t1_val)  + (-3*t1_val-t2_val));
					curr_gbran[0] = k_A*ta_AB + k_AB*(2*(ta_r-ta_AB)-t1_val*th_AB/2) + 2*t2_val*th_r/2;
					curr_gbran[1] = k_B*ta_AB + k_AB*t1_val*th_AB/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h) + k_AB*t1_val*th_AB/2;
					break;
				case 19: //001 ILS ac|b 1th coal_event in branch_AB and 2th coal_event in branch_r
					//curr_gprob = (1-p_g)*(1-p_i)*(p_o) * exp(-2/th_B*(ta_AB-ta_h) + (-2/th_AB*(ta_r-ta_AB)+t1_val)  + (-3*t1_val-t2_val));
					curr_gbran[0] = k_A*ta_AB + k_AB*t1_val*th_AB/2;
					curr_gbran[1] = k_B*ta_AB + k_AB*(2*(ta_r-ta_AB)-t1_val*th_AB/2) + 2*t2_val*th_r/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h) + k_AB*t1_val*th_AB/2;
					break;
				case 20: //001 ILS ab|c all coal_events in branch_r
					C = (1-p_g)*(1-p_i)*(p_o) * C_001;
					//curr_gprob = (1-p_g)*(1-p_i)*(p_o) * exp(-2/th_B*(ta_AB-ta_h) -6/th_AB*(ta_r-ta_AB) -3*t1_val-t2_val);
					curr_gbran[0] = k_A*ta_AB + k_AB*(ta_r-ta_AB) + t1_val*th_r/2;
					curr_gbran[1] = k_B*ta_AB + k_AB*(ta_r-ta_AB) + t1_val*th_r/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h)+ k_AB*(ta_r-ta_AB) + (2*t2_val+t1_val)*th_r/2;
					break;
				case 21: //001 ILS a|bc all coal_events in branch_r
					//curr_gprob = (1-p_g)*(1-p_i)*(p_o) * exp(-2/th_B*(ta_AB-ta_h) -6/th_AB*(ta_r-ta_AB) -3*t1_val-t2_val);
					curr_gbran[0] = k_A*ta_AB + k_AB*(ta_r-ta_AB) + (2*t2_val+t1_val)*th_r/2;
					curr_gbran[1] = k_B*ta_AB + k_AB*(ta_r-ta_AB) + t1_val*th_r/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h) + k_AB*(ta_r-ta_AB) + t1_val*th_r/2;
					break;
				case 22: //001 ILS ac|b all coal_events in branch_r
					//curr_gprob = (1-p_g)*(1-p_i)*(p_o) * exp(-2/th_B*(ta_AB-ta_h) -6/th_AB*(ta_r-ta_AB) -3*t1_val-t2_val);
					curr_gbran[0] = k_A*ta_AB + k_AB*(ta_r-ta_AB) + t1_val*th_r/2;
					curr_gbran[1] = k_B*ta_AB + k_AB*(ta_r-ta_AB) + (2*t2_val+t1_val)*th_r/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h)+ k_AB*(ta_r-ta_AB) + t1_val*th_r/2;
					break;
				//#######################################011##############################
				case 23: //011 ac|b
					curr_gprob = (1-p_g)*p_i*p_o * exp(-t1_val-t2_val);
					curr_gbran[0] = k_A*ta_AB + k_AB*t1_val*th_AB/2;
					curr_gbran[1] = k_B*ta_h + k_C*(ta_r-ta_h) + 2*t2_val*th_r/2 + k_AB*(ta_r-ta_AB-t1_val*th_AB/2);
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h) + k_AB*t1_val*th_AB/2;
					break;
				case 24: //011 ILS ab|c
					C = (1-p_g)*p_i*p_o * C_011;
					//curr_gprob = (1-p_g)*p_i*p_o * exp(-2*(ta_r-ta_AB)/th_AB -3*t1_val-t2_val); 
					curr_gbran[0] = k_A*ta_AB + k_AB*(ta_r-ta_AB) + t1_val*th_r/2;
					curr_gbran[1] = k_B*ta_h + k_C*(ta_r-ta_h) + t1_val*th_r/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h) + k_AB*(ta_r-ta_AB) + (2*t2_val+t1_val)*th_r/2;
					break;
				case 25: //011 ILS a|bc
					//curr_gprob = (1-p_g)*p_i*p_o * exp(-2*(ta_r-ta_AB)/th_AB -3*t1_val-t2_val); 
					curr_gbran[0] = k_A*ta_AB + k_AB*(ta_r-ta_AB) + (2*t2_val+t1_val)*th_r/2;
					curr_gbran[1] = k_B*ta_h + k_C*(ta_r-ta_h) + t1_val*th_r/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h) + k_AB*(ta_r-ta_AB) + t1_val*th_r/2;
					break;
				case 26: //011 ILS b|ac
					//curr_gprob = (1-p_g)*p_i*p_o * exp(-2*(ta_r-ta_AB)/th_AB -3*t1_val-t2_val); 
					curr_gbran[0] = k_A*ta_AB + k_AB*(ta_r-ta_AB) + t1_val*th_r/2;
					curr_gbran[1] = k_B*ta_h + k_C*(ta_r-ta_h) + (2*t2_val+t1_val)*th_r/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h) + k_AB*(ta_r-ta_AB) + t1_val*th_r/2;
					break;
				//#######################################110##############################
				case 27: //110 a|bc 1th coal_event in branch C
					curr_gprob = p_g*p_i*(1-p_o) * exp(-t1_val-t2_val);
					curr_gbran[0] = 2*t2_val*th_r/2 + 2*ta_g-ta_r + k_C*(ta_r-ta_h-t1_val*th_C/2);
					curr_gbran[1] = k_B*ta_h + k_C*t1_val*th_C/2;
					curr_gbran[2] = k_C*(ta_h+t1_val*th_C/2);
					break;
				case 28: //110 a|bc 1th coal_event in branch R
					curr_gprob = p_g*p_i*(1-p_o) * exp(-2*(ta_r-ta_h)/th_C -t1_val-t2_val);
					curr_gbran[0] = ta_g + 2*t2_val*th_r/2 + ta_g-ta_r-t1_val*th_r/2;
					curr_gbran[1] = k_B*ta_h + k_C*(ta_r-ta_h) + t1_val*th_r/2;
					curr_gbran[2] = k_C*ta_r + t1_val*th_r/2;
					break;
				case 29: //110 ILS ab|c
					C =  p_g*p_i*(1-p_o) * C_110;
					//curr_gprob = p_g*p_i*(1-p_o) * exp(-2*(ta_r-ta_h)/th_C -2*(ta_g-ta_r)/th_r -3*t1_val-t2_val);
					curr_gbran[0] = ta_g + t1_val*th_r/2;
					curr_gbran[1] = k_B*ta_h + k_C*(ta_r-ta_h) + ta_g-ta_r + t1_val*th_r/2;
					curr_gbran[2] = k_C*ta_r + ta_g-ta_r + (2*t2_val+t1_val)*th_r/2;
					break;
				case 30: //110 ILS a|bc
					//curr_gprob = p_g*p_i*(1-p_o) * exp(-2*(ta_r-ta_h)/th_C -2*(ta_g-ta_r)/th_r -3*t1_val-t2_val);
					curr_gbran[0] = ta_g + (2*t2_val+t1_val)*th_r/2;
					curr_gbran[1] = k_B*ta_h + k_C*(ta_r-ta_h) + ta_g-ta_r + t1_val*th_r/2;
					curr_gbran[2] = k_C*ta_r + ta_g-ta_r + t1_val*th_r/2;
					break;
				case 31: //110 ILS ac|b
					//curr_gprob = p_g*p_i*(1-p_o) * exp(-2*(ta_r-ta_h)/th_C -2*(ta_g-ta_r)/th_r -3*t1_val-t2_val);
					curr_gbran[0] = ta_g + t1_val*th_r/2;
					curr_gbran[1] = k_B*ta_h + k_C*(ta_r-ta_h) + ta_g-ta_r + (2*t2_val+t1_val)*th_r/2;
					curr_gbran[2] = k_C*ta_r + ta_g-ta_r + t1_val*th_r/2;
					break;
				
				//#######################################101##############################
				case 32: //101 a|bc 1th coal-event in branch B
					curr_gprob = p_g*(1-p_i)*p_o * exp(-t1_val-t2_val);
					curr_gbran[0] = 2*ta_g-ta_r + 2*t2_val*th_r/2 + k_AB*(ta_r-ta_AB) + k_B*(ta_AB-ta_h-t1_val*th_B/2);
					curr_gbran[1] = k_B*ta_h + k_B*t1_val*th_B/2;
					curr_gbran[2] = k_C*ta_h + k_B*t1_val*th_B/2;
					break;
				case 33: //101 a|bc 1th coal-event in branch AB
					curr_gprob = p_g*(1-p_i)*p_o * exp(-2*(ta_AB-ta_h)/th_B - t1_val-t2_val);
					curr_gbran[0] = 2*ta_g-ta_r + 2*t2_val*th_r/2 + k_AB*(ta_r-ta_AB-t1_val*th_AB/2);
					curr_gbran[1] = k_B*ta_AB + k_AB*t1_val*th_AB/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h) + k_AB*t1_val*th_AB/2;
					break;
				case 34: //101 a|bc 1th coal-event in branch R
					curr_gprob = p_g*(1-p_i)*p_o * exp(-2*(ta_AB-ta_h)/th_B-2*(ta_r-ta_AB)/th_AB -t1_val-t2_val);
					curr_gbran[0] = 2*ta_g-ta_r + 2*t2_val*th_r/2 - t1_val*th_r/2;
					curr_gbran[1] = k_B*ta_AB + k_AB*(ta_r-ta_AB) + t1_val*th_r/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h) + k_AB*(ta_r-ta_AB) + t1_val*th_r/2;
					break;
				case 35: //101 ILS ab|c
					C = p_g*(1-p_i)*p_o * C_101;
					//curr_gprob = p_g*(1-p_i)*p_o * exp(-2*(ta_AB-ta_h)/th_B-2*(ta_r-ta_AB)/th_AB-2*(ta_g-ta_r)/th_r - 3*t1_val-t2_val);
					curr_gbran[0] = ta_g + t1_val*th_r/2;
					curr_gbran[1] = k_B*ta_AB + k_AB*(ta_r-ta_AB) + (ta_g-ta_r) + t1_val*th_r/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h) + k_AB*(ta_r-ta_AB) + (ta_g-ta_r) + (2*t2_val+t1_val)*th_r/2;
					break;
				case 36: //101 ILS a|bc
					//curr_gprob = p_g*(1-p_i)*p_o * exp(-2*(ta_AB-ta_h)/th_B-2*(ta_r-ta_AB)/th_AB-2*(ta_g-ta_r)/th_r - 3*t1_val-t2_val);
					curr_gbran[0] = ta_g + (2*t2_val+t1_val)*th_r/2;
					curr_gbran[1] = k_B*ta_AB + k_AB*(ta_r-ta_AB) + (ta_g-ta_r) + t1_val*th_r/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h) + k_AB*(ta_r-ta_AB) + (ta_g-ta_r) + t1_val*th_r/2;
					break;
				case 37: //101 ILS ac|b
					//curr_gprob = p_g*(1-p_i)*p_o * exp(-2*(ta_AB-ta_h)/th_B-2*(ta_r-ta_AB)/th_AB-2*(ta_g-ta_r)/th_r - 3*t1_val-t2_val);
					curr_gbran[0] = ta_g + t1_val*th_r/2;
					curr_gbran[1] = k_B*ta_AB + k_AB*(ta_r-ta_AB) + (ta_g-ta_r) + (2*t2_val+t1_val)*th_r/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h) + k_AB*(ta_r-ta_AB) + (ta_g-ta_r) + t1_val*th_r/2;
					break;
			
				//#######################################111##############################
				case 38: //111 a|bc 
					curr_gprob = p_g*p_i*p_o * exp(-t1_val-t2_val);
					curr_gbran[0] = ta_g + 2*t2_val*th_r/2 + ta_g-ta_r-t1_val*th_r/2;
					curr_gbran[1] = k_B*ta_h + k_C*(ta_r-ta_h) + t1_val*th_r/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h) + k_AB*(ta_r-ta_AB) + t1_val*th_r/2;
					break;
				case 39: //111 ILS ab|c
					C = p_g*p_i*p_o * C_111;
					//curr_gprob = p_g*p_i*p_o * exp(-2/th_r*(ta_g-ta_r) - 3*t1_val-t2_val);
					curr_gbran[0] = ta_g + t1_val*th_r/2;
					curr_gbran[1] = k_B*ta_h + k_C*(ta_r-ta_h) + ta_g-ta_r + t1_val*th_r/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h) + k_AB*(ta_r-ta_AB) + ta_g-ta_r + (2*t2_val+t1_val)*th_r/2;
					break;
				case 40: //111 ILS a|bc
					//curr_gprob = p_g*p_i*p_o * exp(-2/th_r*(ta_g-ta_r) - 3*t1_val-t2_val);
					curr_gbran[0] = ta_g + (2*t2_val+t1_val)*th_r/2;
					curr_gbran[1] = k_B*ta_h + k_C*(ta_r-ta_h) + ta_g-ta_r + t1_val*th_r/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h) + k_AB*(ta_r-ta_AB) + ta_g-ta_r + t1_val*th_r/2;
					break;
				case 41: //111 ILS ac|b
					//curr_gprob = p_g*p_i*p_o * exp(-2/th_r*(ta_g-ta_r) - 3*t1_val-t2_val);
					curr_gbran[0] = ta_g + t1_val*th_r/2;
					curr_gbran[1] = k_B*ta_h + k_C*(ta_r-ta_h) + ta_g-ta_r + (2*t2_val+t1_val)*th_r/2;
					curr_gbran[2] = k_C*ta_h + k_B*(ta_AB-ta_h) + k_AB*(ta_r-ta_AB) + ta_g-ta_r + t1_val*th_r/2;
					break;
			}
///*
			// only for lambda_noP1 = 1
			if (GTREE_IF_ILS_MAP1[gclass]) {
				config.Gprob[1][grid] += config.ILS_ref[grid] * C * weight;
			}else if (GTREE_IF_ILS_MAP2[gclass]) {
				config.Gprob[2][grid] += config.ILS_ref[grid] * C * weight;
			}else if (GTREE_IF_ILS_MAP3[gclass]) {
				config.Gprob[3][grid] += config.ILS_ref[grid] * C * weight;
			}else if (GTREE_IF_ILS_MAP_G1[gclass]) {
				config.Gprob[5][grid] += config.ILS_ref[grid] * C * weight;
			}else if (GTREE_IF_ILS_MAP_G2[gclass]) {
				config.Gprob[6][grid] += config.ILS_ref[grid] * C * weight;
			}else if (GTREE_IF_ILS_MAP_G3[gclass]) {
				config.Gprob[7][grid] += config.ILS_ref[grid] * C * weight;
			}else if (GTREE_EQ_MAP_G[gclass]) {
				config.Gprob[4][grid] += curr_gprob * weight;
			}else {
				config.Gprob[gclass][grid] = curr_gprob*weight;
			}

			if (!GTREE_DELETE[gclass]) LogpFromb(config.Sprob[gclass]+5*grid, curr_gbran);

//*/
			//if (GTREE_IF_ILS_MAP0[gclass]) curr_gprob = config.ILS_ref[grid] * C;
			//config.Gprob[gclass][grid] = curr_gprob*weight;
			//LogpFromb(config.Sprob[gclass]+5*grid, curr_gbran);
		}
	}
}

void get_legendre_points_weights(int n, double nodes[], double weights[])
{
    const gsl_integration_fixed_type *T = gsl_integration_fixed_legendre;
    gsl_integration_fixed_workspace *work =
        gsl_integration_fixed_alloc(T, n, -1.0, 1.0, 0.0, 0.0);
    for (int i=0; i<n; i++) {
        nodes[i]   = work->x[i];
        weights[i] = work->weights[i];
    }
    /* OPT: precompute y_node and weight_base arrays (III.3) */
    for (int i=0; i<n; i++) {
        y_node[i] = (1.0 + nodes[i]) * 0.5;
    }
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            weight_base[i*n+j] = weights[i] * weights[j];
        }
    }
    gsl_integration_fixed_free(work);
}

//   ln(∑grid=i[ ∏ site-pattern=j[pij^nij] * Pgi ])
// = ln(∑i[ exp(∑j(nij*log(pij))) * Pgi ])
// = ln(∑i[ exp(-M + ∑j(nij*log(pij))) * Pgi ]) + M       ##log-sum-exp, 
// The scaling factor M is lnMax=ln[Max{P(D|G)*P(G|M)}]=ln[Max(∑j(nj*log(p))) * Pg)], so the largest term exp(-M + ∑j(nij*log(pij))) * Pgi becomes 1.
double lfun_locus (int locus) {
	double lnL=0, pD=0, test_pg=0, *p, value=0, lmax=data.lnLmax[locus];
	double current_pG_M, current_pD_G, current_pD_G0;
	int gclass, *n, grid, ptree, max_K;
	//printf("f=%-12.6f, locus = %d\n", lmax, locus);

	n = data.Nij + locus * 5;
	for(int g_pos=0; g_pos < config.ngtree; g_pos++) {
		gclass = MODEL_TO_GTREE_MAP[config.model][g_pos];
		if (config.Gprob[gclass][0] == 0) continue;  
		//if (locus <= DBGLOCUS) printf ("%d\n",gclass);
		max_K = (GTREE_IF_ILS_MAP[gclass] ? 3 : 1);

		//Gaussian Integration
		for (grid = 0; grid < m*m; ++grid) {
			// prob of gclass for the given model & update branches: current_b
			current_pG_M = config.Gprob[gclass][grid];
			if (current_pG_M < 1e-100) continue;  
			current_pD_G = 0;
			value = 0;
			// prob of D for the given g, -lnL
			for (int k=0; k<max_K; k++){
				p = config.Sprob[gclass+k] + grid*5;
				// To avoid underflows and overflows, the highest log likelihood at the locus lmax is used for scaling
				current_pD_G0 = -lmax + n[0]*p[0] + n[1]*p[1] + n[2]*p[2] + n[3]*p[3] + n[4]*p[4]; // -lmax > 300
				current_pD_G += (current_pD_G0 < -300 ? 0: exp(current_pD_G0)); 
				//test_pg += current_pG_M;
			}
			value += current_pD_G*current_pG_M;

			pD += value;
			/*
			if (locus <= DBGLOCUS && grid == 5*m+5){
				printf ("n0: %d; n1:%d; n2:%d; n3:%d; n4:%d\n",n[0],n[1],n[2],n[3],n[4]);
				printf ("p0: %f; p1:%f; p2:%f; p3:%f; p4:%f; p0:%f\n",p[0],p[1],p[2],p[3],p[4], exp(config.Sprob[gclass][grid*5]));
				printf ("Gprob: %-10.6f; lnDprob:%10.6f \n",current_pG_M, current_pD_G);
		 	}*/
		}
		if (GTREE_IF_ILS_MAP[gclass]) g_pos += 2;
	}
	lnL = log(pD) + lmax;
	//if (locus <= DBGLOCUS) printf ("##############Prob of gtrees: %-10.6f, %-10.6f\n",test_pg,log(pD));
	return (lnL);
}

// Initialize the parameters of the models
int GetInitials (double x[], double xb[][2])
{
	config.np = 0;
	if (!config.RV) MODEL_to_PARA[config.model][lambda_A] = -1;
	//MODEL_to_PARA[8][MAXPARAMETERS]
	//theta_r, theta_AB, tau_r, tau_AB, tau_g, tau_h, theta_B, theta_C, phi_g, phi_i, phi_o, lambda_A
	//0,       1,        2,     3,      4,     5,     6,       7,       8,     9,     10,    11, 
	for(int i = 0; i < MAXPARAMETERS; i++)  { 
		int label = MODEL_to_PARA[config.model][i];
		if (label >= 0)
		{
			config.np++;
			config.paramnames[label] = PARANAME_MAP[i];
		}
		if (label >= 0 && config.model <= 3){
			switch (i){
				///*// taus for the root node
				case 2:
					x[label] = Tau_int * (0.5+1.5*random());	
					xb[label][0] = LBOUND;
					xb[label][1] = 0.4999999;
					break;
				//*/
				// taus for internal nodes
				case 3:
				case 5:
					x[label] = 0.1 + 0.9*random();	
					xb[label][0] = LBOUND;
					xb[label][1] = 0.999999;
					break;
				// introgression probability
				case 9:
				case 10:
					x[label] = 0.01+0.99*random();	
					xb[label][0] = LBOUND;
					xb[label][1] = 0.999999;
					break;
				// relative rate
				case 11:
					x[label] = 0.8 + 0.4*random();	
					xb[label][0] = 0.25;
					xb[label][1] = 4;
					break;
				default:
					x[label] = 0.0010 + 0.01*random();	
					xb[label][0] = LBOUND;
					xb[label][1] = 0.4999999;
					break;
			}
		}else if (label >= 0 && config.model >= 4 ){
			switch (i){
				case 2:
				case 3:
				case 5:
					x[label] = 0.1 + 0.9*random();	
					xb[label][0] = LBOUND;
					xb[label][1] = 0.999999;
					break;
				case 8:
				case 9:
				case 10:
					x[label] = 0.01 + 0.99*random();	
					xb[label][0] = LBOUND;
					xb[label][1] = 0.999999;
					break;
				case 11:
					x[label] = 0.8 + 0.4*random();	
					xb[label][0] = 0.25;
					xb[label][1] = 4;
					break;
				default:
					x[label] = 0.001 + 0.01*random();	
					xb[label][0] = LBOUND;
					xb[label][1] = 0.4999999;
					break;
			}
		}
	}
	config.ngtree = MODEL_TO_NG_MAP[config.model];
	return(0);
}
//tau_r = tau_g * x1;
//tau_AB = tau_r * x2; 
//tau_h = tau_AB * x3; 
void copyParams(double x[]) 
{
	int i;
	memset(para, 0, MAXPARAMETERS * sizeof(double));
	if (!config.RV) para[lambda_A] = 1;
	para[lambda_B] = para[lambda_AB] = para[lambda_C] = 1;
	for (i = 0; i < MAXPARAMETERS; i++) {
		int target_index = MODEL_to_PARA[config.model][i];
		if(target_index != -1){
			if (x[target_index] != 0) {
				para[i] = x[target_index];
			}else{
				fprintf(stderr, "Error: Parameter count mismatch between array x and the model %d.\n", config.model);
			}
		}
	}
	// the tranformation x=tau1/tau0 is used with the bounds: 0<x<1
	if (config.model == 0){
		para[tau_AB] *= para[tau_r];
	}else if (config.model >= 1 && config.model <= 3){
		para[tau_AB] *= para[tau_r];
		para[tau_h] *= para[tau_AB];
	}else if (config.model == 4){
		para[tau_r] *= para[tau_g];
		para[tau_AB] *= para[tau_r];
	}else{
		para[tau_r] *= para[tau_g];
		para[tau_AB] *= para[tau_r];
		para[tau_h] *= para[tau_AB];
	}
	/*
	printf ("paras: ");
	for (i = 0; i < MAXPARAMETERS; i++) {
		if (para[i]) printf ("%d/%-12.6f ",i,para[i]);
	}
	printf ("\n");*/
}

double lfun(double x[], int np)
{
	double val;
	double Li=0, lnL=0, lnL_sum=0;
	copyParams(x);
	getGtreeIntegrationLimits();
	Cal_Gtree_info ();

//#pragma omp parallel for default(none) reduction(+:lnL, lnL_sum) private(Li) shared(config) num_threads(config.nthreads)
#pragma omp parallel for default(none) reduction(+:lnL) private(Li) shared(config) num_threads(config.nthreads)
	for(int locus=0; locus < config.ndata; locus++)  {
		Li = lfun_locus(locus);
		lnL += Li;
		//lnL_sum += exp(Li);
	}
	/*printf("test_lnL  = %f \n", lnL_sum);	
	printf("lnL  = %f \n", lnL);	
	for (int i = 0; i < MAXPARAMETERS; i++) {
		if (para[i]) fprintf (stderr, "%-12.6f", para[i]);
	}
	fprintf(stderr, "\n");*/
	num_run++;
	//if(num_run%100 == 0) fprintf(stderr,"\nTime used: %s\n", printtime(timestr));
	return(-lnL);
}

/* ================================================================
 *  * Section 5: Model running & LRT
 *   * ================================================================ */
typedef struct {
	int np;
	double lnL;
} ModelResult;
const double chi2CV_5pct[14] = {0, 3.84146, 5.99146, 7.81473, 9.48773, 11.0705, 12.5916, 14.0671, 15.5073, 16.919, 18.307, 19.6751, 21.0261, 22.362 };
const double chi2CV_1pct[14] = { 
	0,
    6.63490,   // df=1
    9.21034,   // df=2
    11.3449,   // df=3
    13.2767,   // df=4
    15.0863,   // df=5
    16.8119,   // df=6
    18.4753,   // df=7
    20.0902,   // df=8
    21.6660,   // df=9
    23.2093,   // df=10
    24.7250,   // df=11
    26.2170,   // df=12
    27.6881    // df=13
};

int getNestedModels(int model, int nested[]) {
    int g = (model >> 2) & 1;
    int i = (model >> 1) & 1;
    int o = model & 1;
    int cnt = 0;
    
    for (int m = 0; m < model; m++) {
        int mg = (m >> 2) & 1;
        int mi = (m >> 1) & 1;
        int mo = m & 1;
        
        if (mg <= g && mi <= i && mo <= o) {
            nested[cnt++] = m;
        }
    }
    return cnt;
}

int RunModel (FILE *f_bfgs, double space[])
{
	int original_RV = config.RV;
	double ln_M0_noRV;
	double x[MAXPARAMETERS]={1,1,1,1,1}, xb[MAXPARAMETERS][2];
	double para_copy[MAXPARAMETERS];
	double lnL=0, lnL0=0, e=1e-8;
	int np;
	int LRT[10] = {0};
	const int MAX_RETRY = 10; 
	const int MAX_run = config.repeat; 
	ModelResult* modelResults = malloc(8 * sizeof(ModelResult));
	enum {M0, M1, M2, M3, M4, M5, M6, M7};
	int runmodels[8] = {0};
	int ranmodels[8] = {1,1,1,1,1,1,1,1};
	int ori_model;
	if (config.run == 0){
		runmodels[config.model] = 1;
		ori_model=config.model;
	}else if (config.run == 1){
		runmodels[M0] = 1;
		runmodels[config.model] = 1;
		ori_model=config.model;
	}else if (config.run == 2){
		runmodels[M0] = 1;
		runmodels[M1] = 1;
		runmodels[M2] = 1;
		runmodels[M4] = 1;
	}

	Model_Gtree();
	for (int model = M0; model <= M7; model++)
	{
		if (runmodels[model] && ranmodels[model])
		{
			ranmodels[model] = 0;
			config.model = model;
			config.event[G] = (config.model >> 2) & 1; 
			config.event[I] = (config.model >> 1) & 1; 
			config.event[O] = config.model & 1;
			
			if (model == 0 && config.RV != original_RV)
			{
				ln_M0_noRV = -1e20; 
				for (int run_idx = 0; run_idx < MAX_run; run_idx++)
				{
					GetInitials(x, xb);
					np = config.np;
					lnL = lfun(x,np);
					ming2(f_bfgs, &lnL, lfun, NULL, x, xb, space, e, np);
					if (ln_M0_noRV < -lnL) {
						ln_M0_noRV = -lnL; 
						memcpy(para_copy, para, sizeof(para));
					}
				}
			}else
			{
			// Define the minimum lnL threshold: the maximum likelihood value of its nested models.
			int nested[8];
			int nCount = getNestedModels(config.model,nested);	
			double maxNestedLnL = -1e20;
			for (int i = 0; i < nCount; i++) {
				int m = nested[i]; 
				if (runmodels[m] && modelResults[m].lnL > maxNestedLnL) {
					maxNestedLnL = modelResults[m].lnL;
				}
			}
			
			// MLE esitmation for the given model
			modelResults[model].lnL = -1e20; 
			for (int run_idx = 0; run_idx < MAX_run; run_idx++)
			{
				int satisfy = 0;
				int retryCount = 0;
				GetInitials(x, xb);
				np = config.np;

				/*//################examine the GetInitials function
				x[0] = 0.0001;
                	        x[1] = 0.0001;
                        	x[2] = 0.02/0.03;
                	        x[3] = 0.0198/0.02;
                        	x[4] = 1.1;
    	                   	x[5] = 0.03;
        	                x[6] = 0.05;*/
    				/*printf("inital x[]:\n");
		    		for (int i = 0; i < config.np; i++) {
					printf("  x[%d] = %.6f\n", i, x[i]);
   				}
   	 			printf("\nboundary xb[][2]:\n");
    				for (int i = 0; i < config.np; i++) {
        				printf("  xb[%d] = [%.6f, %.6f]\n", i, xb[i][0], xb[i][1]);
    				}
    				printf("=====================================\n\n");
				//lnL = lfun(x,np);
				//fprintf(stderr, "Inital lnL  = %f\n", -lnL);*/

				// #############Iterate optimization until the resulting lnL exceeds the lnL values of all previously computed, simpler nested models.
				/*// nlopt
				while (!satisfy && retryCount < MAX_RETRY) {
				    nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS, (unsigned)np);
				    double lb[MAXPARAMETERS], ub[MAXPARAMETERS];
				    for (int i = 0; i < np; i++) {
				        lb[i] = xb[i][0];
				        ub[i] = xb[i][1];
				    }
				    nlopt_set_lower_bounds(opt, lb);
				    nlopt_set_upper_bounds(opt, ub);
				    nlopt_set_ftol_rel(opt, 1e-8);  
				    nlopt_set_xtol_rel(opt, 1e-6);  
				    nlopt_set_maxeval(opt, 5000); 
				    double minf;
				    nlopt_result result = nlopt_optimize(opt, x, &minf);
				    if (result < 0) {
				        fprintf(stderr, "NLopt optimization failed for Model %d! Code: %d\n", config.model, result);
				        lnL = lfun(x, np); 
				    } else {
				        lnL = minf; // nlopt_obj_func
				        lfun(x, np); 
				    }
				    nlopt_destroy(opt);
				    double current_lnL = -lnL;
				    satisfy = 1;
				    if (current_lnL < maxNestedLnL - 1) {
				        satisfy = 0;
				        retryCount++;
				        GetInitials(x, xb);
				    }
				}*/
				
				// BFGS
				while (!satisfy && retryCount < MAX_RETRY) {
					lnL = lfun(x,np);
					ming2(f_bfgs, &lnL, lfun, NULL, x, xb, space, e, np);
					double current_lnL = -lnL;

					satisfy = 1;
					if (current_lnL < maxNestedLnL-1) {
						satisfy = 0;
						retryCount++;
						//fprintf(stderr, "Model%d lnL=%.6f ≤ its nested model lnL=%.6f，rerun ...(%d)\n",
						//	model, current_lnL, maxNestedLnL, retryCount);
						GetInitials(x, xb);
						lnL = lfun(x, np);
					}
				}
				if (retryCount == MAX_RETRY) fprintf(stderr, "Warnings: Repeat%d is trapped in local optimum; results are unreliable...\n", run_idx);
				if (modelResults[model].lnL < -lnL)
				{
					modelResults[model].lnL = -lnL; 
					memcpy(para_copy, para, sizeof(para));
				}
			}
			modelResults[model].np = np;
			}

			//############# Output ##############
			int first = 1;
			fprintf(stderr, "\n### Model %d (", model);
			fprintf(fout, "\n### Model %d (", model);
			for (int event = G; event <= O; event++)
			{
				if (config.event[event] == 1) {
					if (!first){
						fprintf(stderr, " && ");
						fprintf(fout, " && ");
					}
					fprintf(stderr, "%s", event_MAP[event]);
					fprintf(fout, "%s", event_MAP[event]);
					first = 0;
				}
			}
			fprintf(stderr, first ? (config.RV == original_RV? "Species Tree)\n" : "Species Tree && no RV)\n") : ")\n");
			fprintf(fout  , first ? (config.RV == original_RV? "Species Tree)\n" : "Species Tree && no RV)\n") : ")\n");

			fprintf(stderr, "lnL = %-12.6f\n", (config.RV == original_RV ? modelResults[model].lnL : ln_M0_noRV));
			fprintf(stderr, "MLEs:\n\t");
			fprintf(fout, "lnL = %-12.6f\n", (config.RV == original_RV ? modelResults[model].lnL : ln_M0_noRV));
			fprintf(fout, "MLEs:\n\t");
			for(int j=0; j<config.np; j++) 
			{
				if (strstr(config.paramnames[j], "theta") == NULL)
				{
					fprintf(stderr, "%-12s", config.paramnames[j]);
					fprintf(fout, "%-12s", config.paramnames[j]);
				}
			}
			fprintf(stderr, "\n\t");
			fprintf(fout, "\n\t");
			//for (int j = 0; j < config.np; j++) {
			for (int i = 0, j = 0; i < MAXPARAMETERS && j < config.np; i++) {
				if (para_copy[i]) {
					if (strstr(config.paramnames[j], "theta") == NULL) {
						fprintf(stderr, "%-12.6f", para_copy[i]);
						fprintf(fout, "%-12.6f", para_copy[i]);
					}
					j++;
				}
			}

			fprintf(stderr, "\n");
			fprintf(stderr,"Time used: %s\n", printtime(timestr));
			fprintf(fout, "\n");
			fprintf(fout,"Time used: %s\n", printtime(timestr));


			//########## LRT ###########
			if (config.run == 1){
			   if(model == ori_model){
				fprintf (stderr, "\n===== Pairwise LRT: M0 (null)  VS  M%d (introgression) =====\n", model);
				fprintf (fout, "\n===== Pairwise LRT: M0 (null)  VS  M%d (introgression) =====\n", model);
				double lrt = 2 * (modelResults[model].lnL - modelResults[M0].lnL);
				int df = modelResults[model].np - modelResults[M0].np;
				fprintf(stderr, "2DlnL (M%d vs M0) = %f %s chi2CV_1pct[%d] = %f %s\n", model, lrt, (lrt > chi2CV_1pct[df]) ? ">" : "<", df, chi2CV_1pct[df], (lrt > chi2CV_1pct[df]) ? "**" : "ns");
				fprintf(fout  , "2DlnL (M%d vs M0) = %f %s chi2CV_1pct[%d] = %f %s\n", model, lrt, (lrt > chi2CV_1pct[df]) ? ">" : "<", df, chi2CV_1pct[df], (lrt > chi2CV_1pct[df]) ? "**" : "ns");
				if (lrt > chi2CV_1pct[df]){
					fprintf(stderr, "LRT significant (p < 0.01): alternative introgression model M%d is favoured over M0.\n", model);
					fprintf(fout  , "LRT significant (p < 0.01): alternative introgression model M%d is favoured over M0.\n", model);
				}else{
					fprintf(stderr, "LRT non‑significant (p ≥ 0.01): cannot reject null model M0.\n");
					fprintf(fout  , "LRT non‑significant (p ≥ 0.01): cannot reject null model M0.\n");
				}
			   }
			}else if(config.run == 2){
			// Test0: presence or absence of gene flow
			   if (model == M4){
				fprintf (stderr, "\n<<<<<<<<<<< %s >>>>>>>>>>>>\n", "Introgression Presence Testing");
				fprintf (fout, "\n<<<<<<<<<<< %s >>>>>>>>>>>>\n", "Introgression Presence Testing");
				int test_models[] = {M1, M2, M4};
				for (int i = 0; i < 3; i++) {
					int m = test_models[i];
					double lrt = 2 * (modelResults[m].lnL - modelResults[M0].lnL);
					int df = modelResults[m].np - modelResults[M0].np;
					fprintf(stderr, "2DlnL (M%d vs M0) = %f %s chi2CV_1pct[%d] = %f %s\n", m, lrt, (lrt > chi2CV_1pct[df]) ? ">" : "<", df, chi2CV_1pct[df], (lrt > chi2CV_1pct[df]) ? "**" : "ns");
					fprintf(fout  , "2DlnL (M%d vs M0) = %f %s chi2CV_1pct[%d] = %f %s\n", m, lrt, (lrt > chi2CV_1pct[df]) ? ">" : "<", df, chi2CV_1pct[df], (lrt > chi2CV_1pct[df]) ? "**" : "ns");
					if (lrt > chi2CV_1pct[df]) LRT[0] = 1;
				}
				if (LRT[0]) {
					fprintf(stderr, "Significant Introgression detected (p < 0.01)\n");
					fprintf(fout,   "Significant Introgression detected (p < 0.01)\n");
					model = M2;
					runmodels[M3] = 1;
					runmodels[M7] = 1;
				}else {
					fprintf(stderr, "No significant Introgression detected (p < 0.01)\n");
					fprintf(fout,   "No significant Introgression detected (p < 0.01)\n");
					if (config.RV == 1){
						config.RV = 0;
						ranmodels[M0] = 1;
						model = -1;
					}else{
						fprintf(stderr, "\nNo significant introgression detected (p >= 0.01) (M0)\n");
						fprintf(fout, "\nNo significant introgression detected (p >= 0.01) (M0)\n");
						break;
					}
				}
			// presence or absence of rate variation
			}if (model == 0 && config.RV != original_RV){
				double lrt = 2 * (modelResults[M0].lnL - ln_M0_noRV);
				int df = modelResults[M0].np - 4;
				fprintf (stderr, "\n<<<<<<<<<<< %s >>>>>>>>>>>>\n", "Rate Heterogeneity Testing");
				fprintf (fout, "\n<<<<<<<<<<< %s >>>>>>>>>>>>\n", "Rate Heterogeneity Testing");
				fprintf (stderr, "2DlnL (M%d vs M0-noRV) = %f %s chi2CV_1pct[%d] = %f %s\n", M0, lrt, (lrt > chi2CV_1pct[df]) ? ">" : "<", df, chi2CV_1pct[df], (lrt > chi2CV_1pct[df]) ? "**" : "ns");
				fprintf (fout  , "2DlnL (M%d vs M0-noRV) = %f %s chi2CV_1pct[%d] = %f %s\n", M0, lrt, (lrt > chi2CV_1pct[df]) ? ">" : "<", df, chi2CV_1pct[df], (lrt > chi2CV_1pct[df]) ? "**" : "ns");
				if (lrt > chi2CV_1pct[df]) LRT[1] = 1;
				if (LRT[1]) {
					fprintf(stderr, "Significant Rate Variation between sister lineages detected (p < 0.01) (M0)\n");
					fprintf(fout  , "Significant Rate Variation between sister lineages detected (p < 0.01) (M0)\n");
				}else{
						fprintf(stderr, "No significant Rate Variation detected (p >= 0.01)\n");
						fprintf(fout  , "No significant Rate Variation detected (p >= 0.01)\n");
				}
				break;
			// Introgressiont type
			}else if(model == M7){
				fprintf (stderr, "\n<<<<<<<<<<< %s >>>>>>>>>>>>\n", "Introgression Type Testing");
				fprintf (fout, "\n<<<<<<<<<<< %s >>>>>>>>>>>>\n", "Introgression Type Testing");
				int test_models[] = {M3, M4};
				for (int i = 0; i < 2; i++) {
					int m = test_models[i];
					double lrt = 2 * (modelResults[M7].lnL - modelResults[m].lnL);
					int df = modelResults[M7].np - modelResults[m].np;
					fprintf(stderr, "2DlnL (M7 vs M%d) = %f %s chi2CV_1pct[%d] = %f %s\n",  m, lrt, (lrt > chi2CV_1pct[df]) ? ">" : "<", df, chi2CV_1pct[df], (lrt > chi2CV_1pct[df]) ? "**" : "ns");
					fprintf(fout  , "2DlnL (M7 vs M%d) = %f %s chi2CV_1pct[%d] = %f %s\n",  m, lrt, (lrt > chi2CV_1pct[df]) ? ">" : "<", df, chi2CV_1pct[df], (lrt > chi2CV_1pct[df]) ? "**" : "ns");
					if (lrt > chi2CV_1pct[df]) LRT[i+1] = 1;
				}
				int ints = LRT[1]+LRT[2];
				if (ints == 0){
					fprintf(stderr, "Introgression types remain uncertain (p >= 0.01)\nIt is recommended to increase the number of loci to improve the statistical power of the test\n");
					fprintf(fout, "Introgression types remain uncertain (p >= 0.01)\nIt is recommended to increase the number of loci to improve the statistical power of the test\n");
				}
				if (LRT[1]){
					fprintf (stderr, "Ghost introgression (Ghost-->%s) is significantly detected (p < 0.01)", config.s1);
					fprintf (fout, "Ghost introgression (Ghost-->%s) is significantly detected (p < 0.01)", config.s1);
					fprintf(stderr, LRT[2]? "\n" : " (M4)\n");
					fprintf(fout, LRT[2]? "\n" : " (M4)\n");
				}
				if (LRT[2]){
					fprintf (stderr, "Non-sister introgression between %s and %s is significantly detected (p < 0.01)\n", config.s2, config.o);
					fprintf (fout, "Non-sister introgression between %s and %s is significantly detected (p < 0.01)\n", config.s2, config.o);
					model = M4;
					runmodels[M5] = 1;
					runmodels[M6] = 1;
				}
			}else if (model == M6){
				fprintf (stderr, "\n<<<<<<<<<<< %s >>>>>>>>>>>>\n", "Introgression Detection Testing");
				fprintf (fout, "\n<<<<<<<<<<< %s >>>>>>>>>>>>\n", "Introgression Detection Testing");
				int test_models[] = {M5, M6};
				for (int i = 0; i < 2; i++) {
					int m = test_models[i];
					double lrt = 2 * (modelResults[M7].lnL - modelResults[m].lnL);
					int df = modelResults[M7].np - modelResults[m].np;
					fprintf(stderr, "2DlnL (M7 vs M%d) = %f %s chi2CV_1pct[%d] = %f %s\n",  m, lrt, (lrt > chi2CV_1pct[df]) ? ">" : "<", df, chi2CV_1pct[df], (lrt > chi2CV_1pct[df]) ? "**" : "ns");
					fprintf(fout  , "2DlnL (M7 vs M%d) = %f %s chi2CV_1pct[%d] = %f %s\n",  m, lrt, (lrt > chi2CV_1pct[df]) ? ">" : "<", df, chi2CV_1pct[df], (lrt > chi2CV_1pct[df]) ? "**" : "ns");
					if (lrt > chi2CV_1pct[df]) LRT[i+3] = 1;
				}
				int ints = LRT[3]+LRT[4];
				if (ints == 0){
					fprintf(stderr, "Direction of non-sister introgression is unresolved (p >= 0.01)\n");
					fprintf(fout, "Direction of non-sister introgression is unresolved (p >= 0.01)\n");
				}else if (ints==1 && LRT[3]){
					fprintf(stderr, "Inflow (%s<--%s) direction is significantly detected (p < 0.01)\n", config.s2, config.o);
					fprintf(fout, "Inflow (%s<--%s) direction is significantly detected (p < 0.01)\n", config.s2, config.o);
				}else if (ints==1 && LRT[4]){
					fprintf(stderr, "Outflow (%s-->%s) direction is significantly detected (p < 0.01)\n", config.s2, config.o);
					fprintf(fout, "Outflow (%s-->%s) direction is significantly detected (p < 0.01)\n", config.s2, config.o);
				}else{
					fprintf(stderr, "Both inflow and outflow (%s<-->%s) directions are significantly detected (p < 0.01)\n", config.s2, config.o);
					fprintf(fout, "Both inflow and outflow (%s<-->%s) directions are significantly detected (p < 0.01)\n", config.s2, config.o);
				}
				break;
			}
		}}
	}
	//fprintf(stderr,"Time used: %s\n", printtime(timestr));
	free(modelResults);
	return 0;
}


/* ================================================================
 * Section 6: Utility functions
 * ================================================================ */

static time_t time_start;

void starttimer (void)
{
   time_start=time(NULL);
}

char* printtime (char timestr[])
{
/* print time elapsed since last call to starttimer()
*/
   time_t t;
   int h, m, s;

   t = time(NULL)-time_start;
   h = (int)t/3600;
   m = (int)(t%3600)/60;
   s = (int)(t-(t/60)*60);
   if(h)  sprintf(timestr,"%d:%02d:%02d", h,m,s);
   else   sprintf(timestr,"%2d:%02d", m,s);
   return(timestr);
}

double random() 
{
	int val = rand();
	if (val != 0 && val != RAND_MAX){
		return (double) val / RAND_MAX;
    	}else{
		return 0.5;
	}
}
void trim(char *str) {
    char *p = str;
    while (*p && isspace((unsigned char)*p)) p++;
    if (p != str) memmove(str, p, strlen(p) + 1);

    size_t len = strlen(str);
    while (len > 0 && isspace((unsigned char)str[len - 1])) {
        str[len - 1] = '\0';
        len--;
    }
}

void error2 (char * message)
{ fprintf(stderr, "\nError: %s.\n", message); exit(-1); }

int zero (double x[], int n)
{ int i; for(i=0; i<n; i++) x[i]=0; return (0);}

double sum (double x[], int n)
{ int i; double t=0;  for(i=0; i<n; i++) t += x[i];    return(t); }

int fillxc (double x[], double c, int n)
{ int i; for(i=0; i<n; i++) x[i]=c; return (0); }

int xtoy (double x[], double y[], int n)
{ int i; for (i=0; i<n; y[i]=x[i],i++) {}  return(0); }

int abyx (double a, double x[], int n)
{ int i; for (i=0; i<n; x[i]*=a,i++) {}  return(0); }

int axtoy(double a, double x[], double y[], int n)
{ int i; for (i=0; i<n; y[i] = a*x[i],i++) {}  return(0);}

int axbytoz(double a, double x[], double b, double y[], double z[], int n)
{ int i; for(i=0; i<n; i++)   z[i] = a*x[i]+b*y[i];  return (0); }

int identity (double x[], int n)
{ int i,j;  for(i=0; i<n; i++)  { for(j=0; j<n; j++)   x[i*n+j]=0;  x[i*n+i]=1; }  return (0); }

double distance (double x[], double y[], int n)
{  int i; double t=0;
   for (i=0; i<n; i++) t += square(x[i]-y[i]);
   return(sqrt(t));
}

double innerp (double x[], double y[], int n)
{ int i; double t=0;  for(i=0; i<n; i++)  t += x[i]*y[i];  return(t); }

double norm (double x[], int n)
{ int i; double t=0;  for(i=0; i<n; i++)  t += x[i]*x[i];  return sqrt(t); }



/* ================================================================
 * Section 7: BFGS Minimization
 * ================================================================ */

int H_end (double x0[], double x1[], double f0, double f1,
    double e1, double e2, int n)
/*   Himmelblau termination rule.   return 1 for stop, 0 otherwise.
*/
{
   double r;
   if((r=norm(x0,n))<e2)
      r=1;
   r*=e1;
   if(distance(x1,x0,n)>=r)
      return(0);
   r=fabs(f0);  if(r<e2) r=1;     
   r*=e1;
   if(fabs(f1-f0)>=r) 
      return(0);
   return (1);
}

double Small_Diff=0.5e-9;  /* reasonable values 1e-5, 1e-7 */
//double Small_Diff=0.5e-11;  /* reasonable values 1e-5, 1e-7 */

int gradientB(int n, double x[], double f0, double g[],
              double (*fun)(double x[],int n),
              double space[], int xmark[])
{
    int i, j;
    double *x0 = space, *x1 = space + n;
    /* OPT: optimal FD step constants */
    const double cbrt_eps = cbrt(DBL_EPSILON);  /* ~6e-6 */
    const double sqrt_eps = sqrt(DBL_EPSILON);   /* ~1.5e-8 */
    for (i=0; i<n; i++) {
        double ax = fabs(x[i]);
        double scale = (ax > 1.0) ? ax : 1.0;
        if (xmark[i] == 0 && (AlwaysCenter || SIZEp < 1)) {
            /* central difference */
            double eh = cbrt_eps * scale;
            for (j=0; j<n; j++) x0[j] = x1[j] = x[j];
            x0[i] -= eh; x1[i] += eh;
            g[i] = ((*fun)(x1,n) - (*fun)(x0,n)) / (eh * 2.0);
        } else {
            /* forward / backward difference */
            double eh = sqrt_eps * scale;
            for (j=0; j<n; j++) x1[j] = x[j];
            if (xmark[i]) eh *= -xmark[i];
            x1[i] += eh;
            g[i] = ((*fun)(x1,n) - f0) / eh;
        }
    }
    return 0;
}


double fun_LineSearch (double t, double (*fun)(double x[],int n), 
       double x0[], double p[], double x[], int n)
{	
	int i;   
	FOR (i,n) x[i]=x0[i] + t*p[i];
	return( (*fun)(x, n) ); 
}

double LineSearch2 (double(*fun)(double x[],int n), double *f, double x0[], 
       double p[], double step, double limit, double e, double space[], int n)
{
/* linear search using quadratic interpolation 
   from x0[] in the direction of p[],
                x = x0 + a*p        a ~(0,limit)
   returns (a).    *f: f(x0) for input and f(x) for output

   x0[n] x[n] p[n] space[n]

   adapted from Wolfe M. A.  1978.  Numerical methods for unconstrained
   optimization: An introduction.  Van Nostrand Reinhold Company, New York.
   pp. 62-73.
   step is used to find the bracket and is increased or reduced as necessary, 
   and is not terribly important.
*/
   int ii=0, maxround=10, status, i, nsymb=0;
   double *x=space, factor=4, small=1e-10, smallgapa=0.2;
   double a0,a1,a2,a3,a4=-1,a5,a6, f0,f1,f2,f3,f4=-1,f5,f6;

/* look for bracket (a1, a2, a3) with function values (f1, f2, f3)
   step length step given, and only in the direction a>=0
*/

   if (noisy>2)
      fprintf (stderr,"\n%3d h-m-p %7.4f %6.4f %8.4f ",Iround+1,step,limit,norm(p,n));

   if (step<=0 || limit<small || step>=limit) {
      if (noisy>2) 
         printf ("\nh-m-p:%20.8e%20.8e%20.8e %12.6f\n",step,limit,norm(p,n),*f);
      return (0);
   }

   // constructe the initial bracket {x1<x2<x3}, making f1>f2<f3
   a0=a1=0; f1=f0=*f;
   a2=a0+step; f2=fun_LineSearch(a2, fun,x0,p,x,n);
   if (f2>f1) {  /* reduce step length so the algorithm is decreasing */
      for (; ;) {
         step/=factor;
         if (step<small) return (0);
         a3=a2;    f3=f2;
         a2=a0+step;  f2=fun_LineSearch(a2, fun,x0,p,x,n);
         if (f2<=f1) break;
         if(!PAML_RELEASE && noisy>2) { printf("-"); nsymb++; }
      }
   }
   else {       /* step length is too small? */
      for (; ;) {
         step*=factor;
         if (step>limit) step=limit;
         a3=a0+step;  f3=fun_LineSearch(a3, fun,x0,p,x,n);
         if (f3>=f2) break;

         if(!PAML_RELEASE && noisy>2) { printf("+"); nsymb++; }
         a1=a2; f1=f2;    a2=a3; f2=f3;
         if (step>=limit) {
            if(!PAML_RELEASE && noisy>2) for(; nsymb<5; nsymb++) printf(" ");
            if (noisy>2) printf(" %12.6f%3c %6.4f %5d", *f=f3, 'm', a3, NFunCall);
            *f=f3; return(a3);
         }
      }
   }
   //printf (" %8.4f:%8.4f|%8.4f:%8.4f|%8.4f:%8.4f ",a1,f1,a2,f2,a3,f3);

   /* iteration by quadratic interpolation, fig 2.2.9-10 (pp 71-71) */
   for (ii=0; ii<maxround; ii++) {
      /* a4 is the minimum from the parabola over (a1,a2,a3)  */
      a4 = (a2-a3)*f1+(a3-a1)*f2+(a1-a2)*f3;
      if(fabs(a4)>1e-100) 
         a4 = ((a2*a2-a3*a3)*f1+(a3*a3-a1*a1)*f2+(a1*a1-a2*a2)*f3)/(2*a4);
      if (a4>a3 || a4<a1) {   /* out of range */
         a4=(a1+a2)/2;
         status='N';
      }
      else {
         if((a4<=a2 && a2-a4>smallgapa*(a2-a1)) || (a4>a2 && a4-a2>smallgapa*(a3-a2)))
            status='Y';
         else 
            status='C';
      }
      f4 = fun_LineSearch(a4, fun,x0,p,x,n);
      if(!PAML_RELEASE && noisy>2) putchar(status);
      // stop 
      if (fabs(f2-f4)<e*(1+fabs(f2))) {
         if(!PAML_RELEASE && noisy>2) 
            for(nsymb+=ii+1; nsymb<5; nsymb++) printf(" ");
         break;
      }
      //if(fabs(a2-a4)<smallgapa*(a3-a1) || fabs(a1-a4)<smallgapa*(a3-a1) || fabs(a3-a4)<smallgapa*(a3-a1)) printf ("\n###odd: "); 
      //printf ("%8.4f:%8.4f|%8.4f:%8.4f|%8.4f:%8.4f|%8.4f:%8.4f => ",a1,f1,a2,f2,a3,f3,a4,f4);

      /* possible multiple local optima during line search */
      if(!PAML_RELEASE  && noisy>2 && ((a4<a2&&f4>f1) || (a4>a2&&f4>f3))) {
         printf("\n\na %12.6f %12.6f %12.6f %12.6f",   a1,a2,a3,a4);
         printf(  "\nf %12.6f %12.6f %12.6f %12.6f\n", f1,f2,f3,f4);

         for(a5=a1; a5<=a3; a5+=(a3-a1)/20) {
            printf("\t%.6e ",a5);
            if(n<5) FOR(i,n) printf("\t%.6f",x0[i] + a5*p[i]);
            printf("\t%.6f\n", fun_LineSearch(a5, fun,x0,p,x,n));
         }
         puts("Linesearch2 a4: multiple optima?");
      }
      
      if (a4<=a2) {    // fig 2.2.10 
         if (a2-a4>smallgapa*(a2-a1)) {
            if (f4<=f2) { a3=a2; a2=a4;  f3=f2; f2=f4; }
            else        { a1=a4; f1=f4; }
         }
         else {
            if (f4>f2) {
               a5=(a2+a3)/2; f5=fun_LineSearch(a5, fun,x0,p,x,n);
               if (f5>f2) { a1=a4; a3=a5;  f1=f4; f3=f5; }
               else       { a1=a2; a2=a5;  f1=f2; f2=f5; }
            }
            else {
               a5=(a1+a4)/2; f5=fun_LineSearch(a5, fun,x0,p,x,n);
               if (f5>=f4)
                  { a3=a2; a2=a4; a1=a5;  f3=f2; f2=f4; f1=f5; }
               else {
                  a6=(a1+a5)/2; f6=fun_LineSearch(a6, fun,x0,p,x,n);
                  if (f6>f5)
                       { a1=a6; a2=a5; a3=a4;  f1=f6; f2=f5; f3=f4; }
                  else { a2=a6; a3=a5; f2=f6; f3=f5; }
               }
            }
         }
      }
      else {                     // fig 2.2.9 
         if (a4-a2>smallgapa*(a3-a2)) {
            if (f2>=f4) { a1=a2; a2=a4;  f1=f2; f2=f4; }
            else        { a3=a4; f3=f4; }
         }
         else {
            if (f4>f2) {
               a5=(a1+a2)/2; f5=fun_LineSearch(a5, fun,x0,p,x,n);
               if (f5>f2) { a1=a5; a3=a4;  f1=f5; f3=f4; }
               else       { a3=a2; a2=a5;  f3=f2; f2=f5; }
            }
            else {
               a5=(a3+a4)/2; f5=fun_LineSearch(a5, fun,x0,p,x,n);
               if (f5>=f4)
                  { a1=a2; a2=a4; a3=a5;  f1=f2; f2=f4; f3=f5; }
               else {
                  a6=(a3+a5)/2; f6=fun_LineSearch(a6, fun,x0,p,x,n);
                  if (f6>f5)
                      { a1=a4; a2=a5; a3=a6;  f1=f4; f2=f5; f3=f6; }
                  else { a1=a5; a2=a6;  f1=f5; f2=f6; }
               }
            }
         }
      }
      //printf("%6.4f %6.4f %6.4f ", a1,a2,a3);
   }

   if (f2>f0 && f4>f0)  a4=0;
   if (f2<=f4)  { *f=f2; a4=a2; }
   else         *f=f4;
   if(noisy>2) printf(" %12.6f%3d %6.4f %5d", *f, ii, a4, NFunCall);

   return (a4);
}

extern int noisy, Iround;
extern double SIZEp;



#define BFGS
/*
#define SR1
#define DFP
*/

extern FILE *frst;

int ming2 (FILE *fout, double *f, double (*fun)(double x[], int n),
    int (*dfun)(double x[], double *f, double dx[], int n),
    double x[], double xb[][2], double space[], double e, int n)
{
/* n-variate minimization with bounds using the BFGS algorithm
     g0[n] g[n] p[n] x0[n] y[n] s[n] z[n] H[n*n] C[n*n] tv[2*n]
     xmark[n],ix[n]
   Size of space should be (check carefully?)
      #define spaceming2(n) ((n)*((n)*2+9+2)*sizeof(double))
   nfree: # free variables
   xmark[i]=0 for inside space; -1 for lower boundary; 1 for upper boundary.
   x[] has initial values at input and returns the estimates in return.
   ix[i] specifies the i-th free parameter

*/
   int i,j, i1,i2,it, maxround=10000, fail=0, *xmark, *ix, nfree;
   int Ngoodtimes=2, goodtimes=0, reset_counter=0;
   double small=1.e-30, sizep0=0;     /* small value for checking |w|=0 */
   double f0, *g0, *g, *p, *x0, *y, *s, *z, *H, *C, *tv;
   double w,v, alpha, am, h, maxstep=8;

   if(n==0) return(0);
   g0=space;   g=g0+n;  p=g+n;   x0=p+n;
   y=x0+n;     s=y+n;   z=s+n;   H=z+n;  C=H+n*n, tv=C+n*n;
   xmark=(int*)(tv+2*n);  ix=xmark+n;

   // Initialize boundary parameters xmark
   for(i=0; i<n; i++)  { xmark[i]=0; ix[i]=i; }
   for(i=0,nfree=0;i<n;i++) {
      if(x[i]<=xb[i][0]) { x[i]=xb[i][0]; xmark[i]=-1; continue; }
      if(x[i]>=xb[i][1]) { x[i]=xb[i][1]; xmark[i]= 1; continue; }
      ix[nfree++]=i;
   }
   if(noisy>2 && nfree<n && n<50) {
      FPN(F0);  FOR(j,n) printf(" %9.6f", x[j]);  FPN(F0);
      FOR(j,n) printf(" %9.5f", xb[j][0]);  FPN(F0);
      FOR(j,n) printf(" %9.5f", xb[j][1]);  FPN(F0);
      if(nfree<n && noisy>=3) printf("warning: ming2, %d paras at boundary.",n-nfree);
   }

   f0=*f=(*fun)(x,n);
   xtoy(x,x0,n);
   SIZEp=99;
   if (noisy>2) {
      printf ("\nIterating by ming2\nInitial: fx= %12.6f\nx=",f0);
      FOR(i,n) printf(" %8.5f", x[i]);   FPN(F0);
   }
   
   // Initialize the gradient parameter g0
   if (dfun)  (*dfun) (x0, &f0, g0, n);
   else       gradientB (n, x0, f0, g0, fun, tv, xmark);

   // Initialize the Hessenberg inverse matrix H = identity matrix // nfree: the number of parameters not at the boundary
   identity (H,nfree);

   // ############# Enter the main iteration loop of BFGS #################
   for(Iround=0; Iround<maxround; Iround++) {
      if (fout) {
         fprintf (fout, "\n%3d %7.4f %13.6f  x: ", Iround,sizep0,f0);
         FOR (i,n) fprintf (fout, "%8.5f  ", x0[i]);
         fflush (fout);
      }

      // update p
      for (i=0,zero(p,n); i<nfree; i++)  FOR (j,nfree)
         p[ix[i]] -= H[i*nfree+j]*g0[ix[j]];
      sizep0 = SIZEp; 
      SIZEp  = norm(p,n);      /* check this */

      for (i=0,am=maxstep; i<n; i++) {  /* max step length */ //Avoid paras going beyond the boundaries
         if (p[i]>0 && (xb[i][1]-x0[i])/p[i]<am) am=(xb[i][1]-x0[i])/p[i];
         else if (p[i]<0 && (xb[i][0]-x0[i])/p[i]<am) am=(xb[i][0]-x0[i])/p[i];
      }

      if (Iround==0) {
         h=fabs(2*f0*.01/innerp(g0,p,n));  /* check this?? */
         //printf ("\nstep:%3d %12.4f\n", h); 
         h=min2(h,am/2000);
         //printf ("\nstep:%3d %12.4f\n", h); 
      }else {
      // ref to the last step size
         h=norm(s,nfree)/SIZEp;
         h=max2(h,am/500);
      }
      h = max2(h,1e-5);   h = min2(h,am/5);
      *f = f0;
      // find best step
      alpha = LineSearch2(fun,f,x0,p,h,am, min2(1e-3,e), tv,n); 

      if (isnan(*f) || isnan(alpha) || alpha<=0) {
	 reset_counter++;
	 if (reset_counter >= 50) { // 累计10次直接终止
        	Iround = maxround;
        	break;
   	 }
         if (fail) {
            if (AlwaysCenter) { Iround=maxround;  break; }
            else { AlwaysCenter=1; identity(H,n); fail=1; }
         }
         else   
            { if(noisy>2) printf(".. ");  identity(H,nfree); fail=1; }
      }
      // update x && check converge
      else  {
         fail=0;
         FOR(i,n)  x[i]=x0[i]+alpha*p[i];
         w=min2(2,e*1000); if(e<1e-4 && e>1e-6) w=0.01;

         if(Iround==0 || SIZEp<sizep0 || (SIZEp<.001 && sizep0<.001)) goodtimes++;
         else  goodtimes=0;
         if((n==1||goodtimes>=Ngoodtimes) && SIZEp<(e>1e-5?1:.001)
            && H_end(x0,x,f0,*f,e,e,n))
            break;
      }
      // update g
      if (dfun)
         (*dfun) (x, f, g, n);
      else
         gradientB (n, x, *f, g, fun, tv, xmark);
/*
for(i=0; i<n; i++) fprintf(frst,"%9.5f", x[i]); fprintf(frst, "%6d",AlwaysCenter);
for(i=0; i<n; i++) fprintf(frst,"%9.2f", g[i]); FPN(frst);
	*/
      /* modify the working set */
      for(i=0; i<n; i++) {         /* add constraints, reduce H */
         if (xmark[i]) continue;
         if (fabs(x[i]-xb[i][0])<1e-6 && g[i]>0)  xmark[i]=-1;  /*add constraints for new paras reaching upper B */
         else if (fabs(x[i]-xb[i][1])<1e-6 && g[i]<0)  xmark[i]=1;  /*add constraints for new paras reaching lower B */
         if (xmark[i]==0) continue;
         xtoy (H, C, nfree*nfree);
         for(it=0; it<nfree; it++) if (ix[it]==i) break;
         for (i1=it; i1<nfree-1; i1++) ix[i1]=ix[i1+1];
         for (i1=0,nfree--; i1<nfree; i1++) FOR (i2,nfree)
            H[i1*nfree+i2]=C[(i1+(i1>=it))*(nfree+1) + i2+(i2>=it)];
      }
      for (i=0,it=0,w=0; i<n; i++) {  /* delete a constraint, enlarge H */ 
         if (xmark[i]==-1 && -g[i]>w)     { it=i; w=-g[i]; }  //
         else if (xmark[i]==1 && -g[i]<-w) { it=i; w=g[i]; }
      }
      if (w>10*SIZEp/nfree) {          /* *** */
         xtoy (H, C, nfree*nfree);
         FOR (i1,nfree) FOR (i2,nfree) H[i1*(nfree+1)+i2]=C[i1*nfree+i2];
         FOR (i1,nfree+1) H[i1*(nfree+1)+nfree]=H[nfree*(nfree+1)+i1]=0;
         H[(nfree+1)*(nfree+1)-1]=1;
         xmark[it]=0;   ix[nfree++]=it;
      }

      if (noisy>2) {
         printf (" | %d/%d", n-nfree, n);
         /* FOR (i,n)  if (xmark[i]) printf ("%4d", i+1); */
      }

      // update y and s
      for (i=0,f0=*f; i<nfree; i++)
        {  y[i]=g[ix[i]]-g0[ix[i]];  s[i]=x[ix[i]]-x0[ix[i]]; }
      FOR (i,n) { g0[i]=g[i]; x0[i]=x[i]; }

      // update H
      /* renewal of H varies with different algorithms   */
#if (defined SR1)
      /*   Symmetrical Rank One (Broyden, C. G., 1967) */
      for (i=0,w=.0; i<nfree; i++) {
         for (j=0,v=.0; j<nfree; j++) v += H[i*nfree+j] * y[j];
         z[i]=s[i] - v;
         w += y[i]*z[i];
      }
      if (fabs(w)<small)   { identity(H,nfree); fail=1; continue; }
      FOR (i,nfree)  FOR (j,nfree)  H[i*nfree+j] += z[i]*z[j]/w;
#elif (defined DFP)
      /* Davidon (1959), Fletcher and Powell (1963). */
      for (i=0,w=v=0.; i<nfree; i++) {
         for (j=0,z[i]=0; j<nfree; j++) z[i] += H[i*nfree+j] * y[j];
         w += y[i]*z[i];  v += y[i]*s[i];
      }
      if (fabs(w)<small || fabs(v)<small)  { identity(H,nfree); fail=1; continue;}
      FOR (i,nfree)  FOR (j,nfree)  
         H[i*nfree+j] += s[i]*s[j]/v - z[i]*z[j]/w;
#else /* BFGS */
      for(i=0,w=v=0.; i<nfree; i++) {
         for(j=0,z[i]=0.; j<nfree; j++) z[i] += H[i*nfree+j]*y[j];
         w += y[i]*z[i];    v += y[i]*s[i];
      }
      if (fabs(v)<small)   { 
	reset_counter++; 
    	if (reset_counter >= 50) { 
        	Iround = maxround;
        	break;
    	}
	identity(H,nfree); 
	fail=1; 
	continue; 
      }
      double inv_v = 1.0 / v;
      double coeff = (1.0 + w*inv_v) * inv_v;
      for (i=0; i<nfree; i++) {
           for (j=i; j<nfree; j++) {
                double upd = coeff * s[i]*s[j]
                               - inv_v * (z[i]*s[j] + s[i]*z[j]);
                H[i*nfree+j] += upd;
                if (i != j) H[j*nfree+i] += upd;  /* mirror */
            }
       }

      //FOR (i,nfree)  FOR (j,nfree)
      //H[i*nfree+j] += ((1+w/v)*s[i]*s[j]-z[i]*s[j]-s[i]*z[j])/v;
#endif
   }    /* for (Iround,maxround)  */

   /* try to remove this after updating LineSearch2() */
   //*f = (*fun)(x,n);
   if(noisy>2) FPN(F0);

   if(Iround==maxround) {
      if (fout) fprintf (fout,"\ncheck convergence!\n");
      return(-1);
   }
   if(nfree==n) { 
      xtoy(H, space, n*n);  /* H has variance matrix, or inverse of Hessian */
      return(1);
   }
   return(0);
}


/* ================================================================
 * NLopt Objective Function Wrapper
 * ================================================================ */
static double nlopt_obj_func(unsigned n, const double *x, double *grad, void *my_func_data)
{
    int np = (int)n;
    double fcurrent = lfun((double*)x, np); 
    if (grad) {
        double *x_tmp = (double*)malloc(n * sizeof(double));
        if (!x_tmp) return fcurrent; 
        memcpy(x_tmp, x, n * sizeof(double));
        const double cbrt_eps = cbrt(DBL_EPSILON); 
        double h;
        for (unsigned i = 0; i < n; i++) {
            double ax = fabs(x[i]);
            double scale = (ax > 1.0) ? ax : 1.0;
            h = cbrt_eps * scale;
            x_tmp[i] = x[i] + h;
            double f_plus = lfun(x_tmp, np);
            x_tmp[i] = x[i] - h;
            double f_minus = lfun(x_tmp, np);
            x_tmp[i] = x[i];
            grad[i] = (f_plus - f_minus) / (2.0 * h);
        }
        free(x_tmp);
    }
    return fcurrent;
}
