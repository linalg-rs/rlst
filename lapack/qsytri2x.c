#define len_trim__(cad,len) ({                           integer _r=0,i;                           for(i=0; i<(len) && (cad)[i]; i++)                             if((cad)[i] != ' ') _r=i;                           _r+1; })
#define ceiling_(a) (myceil(*(a)))
#define myceil(a) (sizeof(a) == sizeof(float) ? ceilf(a) : ceil(a))
#include <math.h>
/*  -- translated by f2c (version 20200916).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#define __LAPACK_PRECISION_QUAD
#include "f2c.h"

/* Table of constant values */

static quadreal c_b11 = 1.;
static quadreal c_b15 = 0.;

/* > \brief \b DSYTRI2X */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DSYTRI2X + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dsytri2
x.f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dsytri2
x.f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dsytri2
x.f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DSYTRI2X( UPLO, N, A, LDA, IPIV, WORK, NB, INFO ) */

/*       CHARACTER          UPLO */
/*       INTEGER            INFO, LDA, N, NB */
/*       INTEGER            IPIV( * ) */
/*       DOUBLE PRECISION   A( LDA, * ), WORK( N+NB+1,* ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DSYTRI2X computes the inverse of a doublereal symmetric indefinite matrix */
/* > A using the factorization A = U*D*U**T or A = L*D*L**T computed by */
/* > DSYTRF. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          Specifies whether the details of the factorization are stored */
/* >          as an upper or lower triangular matrix. */
/* >          = 'U':  Upper triangular, form is A = U*D*U**T; */
/* >          = 'L':  Lower triangular, form is A = L*D*L**T. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is DOUBLE PRECISION array, dimension (LDA,N) */
/* >          On entry, the NNB diagonal matrix D and the multipliers */
/* >          used to obtain the factor U or L as computed by DSYTRF. */
/* > */
/* >          On exit, if INFO = 0, the (symmetric) inverse of the original */
/* >          matrix.  If UPLO = 'U', the upper triangular part of the */
/* >          inverse is formed and the part of A below the diagonal is not */
/* >          referenced; if UPLO = 'L' the lower triangular part of the */
/* >          inverse is formed and the part of A above the diagonal is */
/* >          not referenced. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in] IPIV */
/* > \verbatim */
/* >          IPIV is INTEGER array, dimension (N) */
/* >          Details of the interchanges and the NNB structure of D */
/* >          as determined by DSYTRF. */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is DOUBLE PRECISION array, dimension (N+NB+1,NB+3) */
/* > \endverbatim */
/* > */
/* > \param[in] NB */
/* > \verbatim */
/* >          NB is INTEGER */
/* >          Block size */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0: successful exit */
/* >          < 0: if INFO = -i, the i-th argument had an illegal value */
/* >          > 0: if INFO = i, D(i,i) = 0; the matrix is singular and its */
/* >               inverse could not be computed. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date June 2017 */

/* > \ingroup doubleSYcomputational */

/*  ===================================================================== */
void  qsytri2x_(char *uplo, integer *n, quadreal *a, integer 
	*lda, integer *ipiv, quadreal *work, integer *nb, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, work_dim1, work_offset, i__1, i__2, i__3;

    /* Local variables */
    extern void  qsyswapr_(char *, integer *, quadreal *, 
	    integer *, integer *, integer *);
    quadreal d__;
    integer i__, j, k;
    quadreal t, ak;
    integer u11, ip, nnb, cut;
    quadreal akp1;
    integer invd;
    quadreal akkp1;
    extern void  qgemm_(char *, char *, integer *, integer *, 
	    integer *, quadreal *, quadreal *, integer *, quadreal *, 
	    integer *, quadreal *, quadreal *, integer *);
    extern logical lsame_(char *, char *);
    integer iinfo;
    extern void  qtrmm_(char *, char *, char *, char *, 
	    integer *, integer *, quadreal *, quadreal *, integer *, 
	    quadreal *, integer *);
    integer count;
    logical upper;
    quadreal u01_i_j__, u11_i_j__;
    extern void  xerbla_(char *, integer *), qtrtri_(
	    char *, char *, integer *, quadreal *, integer *, integer *), qsyconv_(char *, char *, integer *, quadreal *,
	     integer *, integer *, quadreal *, integer *);
    quadreal u01_ip1_j__, u11_ip1_j__;


/*  -- LAPACK computational routine (version 3.7.1) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     June 2017 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;
    work_dim1 = *n + *nb + 1;
    work_offset = 1 + work_dim1;
    work -= work_offset;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -4;
    }

/*     Quick return if possible */


    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DSYTRI2X", &i__1);
	return;
    }
    if (*n == 0) {
	return;
    }

/*     Convert A */
/*     Workspace got Non-diag elements of D */

    qsyconv_(uplo, "C", n, &a[a_offset], lda, &ipiv[1], &work[work_offset], &
	    iinfo);

/*     Check that the diagonal matrix D is nonsingular. */

    if (upper) {

/*        Upper triangular storage: examine D from bottom to top */

	for (*info = *n; *info >= 1; --(*info)) {
	    if (ipiv[*info] > 0 && a[*info + *info * a_dim1] == 0.) {
		return;
	    }
	}
    } else {

/*        Lower triangular storage: examine D from top to bottom. */

	i__1 = *n;
	for (*info = 1; *info <= i__1; ++(*info)) {
	    if (ipiv[*info] > 0 && a[*info + *info * a_dim1] == 0.) {
		return;
	    }
	}
    }
    *info = 0;

/*  Splitting Workspace */
/*     U01 is a block (N,NB+1) */
/*     The first element of U01 is in WORK(1,1) */
/*     U11 is a block (NB+1,NB+1) */
/*     The first element of U11 is in WORK(N+1,1) */
    u11 = *n;
/*     INVD is a block (N,2) */
/*     The first element of INVD is in WORK(1,INVD) */
    invd = *nb + 2;
    if (upper) {

/*        invA = P * inv(U**T)*inv(D)*inv(U)*P**T. */

	qtrtri_(uplo, "U", n, &a[a_offset], lda, info);

/*       inv(D) and inv(D)*inv(U) */

	k = 1;
	while(k <= *n) {
	    if (ipiv[k] > 0) {
/*           1 x 1 diagonal NNB */
		work[k + invd * work_dim1] = 1. / a[k + k * a_dim1];
		work[k + (invd + 1) * work_dim1] = 0.;
		++k;
	    } else {
/*           2 x 2 diagonal NNB */
		t = work[k + 1 + work_dim1];
		ak = a[k + k * a_dim1] / t;
		akp1 = a[k + 1 + (k + 1) * a_dim1] / t;
		akkp1 = work[k + 1 + work_dim1] / t;
		d__ = t * (ak * akp1 - 1.);
		work[k + invd * work_dim1] = akp1 / d__;
		work[k + 1 + (invd + 1) * work_dim1] = ak / d__;
		work[k + (invd + 1) * work_dim1] = -akkp1 / d__;
		work[k + 1 + invd * work_dim1] = -akkp1 / d__;
		k += 2;
	    }
	}

/*       inv(U**T) = (inv(U))**T */

/*       inv(U**T)*inv(D)*inv(U) */

	cut = *n;
	while(cut > 0) {
	    nnb = *nb;
	    if (cut <= nnb) {
		nnb = cut;
	    } else {
		count = 0;
/*             count negative elements, */
		i__1 = cut;
		for (i__ = cut + 1 - nnb; i__ <= i__1; ++i__) {
		    if (ipiv[i__] < 0) {
			++count;
		    }
		}
/*             need a even number for a clear cut */
		if (count % 2 == 1) {
		    ++nnb;
		}
	    }
	    cut -= nnb;

/*          U01 Block */

	    i__1 = cut;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = nnb;
		for (j = 1; j <= i__2; ++j) {
		    work[i__ + j * work_dim1] = a[i__ + (cut + j) * a_dim1];
		}
	    }

/*          U11 Block */

	    i__1 = nnb;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		work[u11 + i__ + i__ * work_dim1] = 1.;
		i__2 = i__ - 1;
		for (j = 1; j <= i__2; ++j) {
		    work[u11 + i__ + j * work_dim1] = 0.;
		}
		i__2 = nnb;
		for (j = i__ + 1; j <= i__2; ++j) {
		    work[u11 + i__ + j * work_dim1] = a[cut + i__ + (cut + j) 
			    * a_dim1];
		}
	    }

/*          invD*U01 */

	    i__ = 1;
	    while(i__ <= cut) {
		if (ipiv[i__] > 0) {
		    i__1 = nnb;
		    for (j = 1; j <= i__1; ++j) {
			work[i__ + j * work_dim1] = work[i__ + invd * 
				work_dim1] * work[i__ + j * work_dim1];
		    }
		    ++i__;
		} else {
		    i__1 = nnb;
		    for (j = 1; j <= i__1; ++j) {
			u01_i_j__ = work[i__ + j * work_dim1];
			u01_ip1_j__ = work[i__ + 1 + j * work_dim1];
			work[i__ + j * work_dim1] = work[i__ + invd * 
				work_dim1] * u01_i_j__ + work[i__ + (invd + 1)
				 * work_dim1] * u01_ip1_j__;
			work[i__ + 1 + j * work_dim1] = work[i__ + 1 + invd * 
				work_dim1] * u01_i_j__ + work[i__ + 1 + (invd 
				+ 1) * work_dim1] * u01_ip1_j__;
		    }
		    i__ += 2;
		}
	    }

/*        invD1*U11 */

	    i__ = 1;
	    while(i__ <= nnb) {
		if (ipiv[cut + i__] > 0) {
		    i__1 = nnb;
		    for (j = i__; j <= i__1; ++j) {
			work[u11 + i__ + j * work_dim1] = work[cut + i__ + 
				invd * work_dim1] * work[u11 + i__ + j * 
				work_dim1];
		    }
		    ++i__;
		} else {
		    i__1 = nnb;
		    for (j = i__; j <= i__1; ++j) {
			u11_i_j__ = work[u11 + i__ + j * work_dim1];
			u11_ip1_j__ = work[u11 + i__ + 1 + j * work_dim1];
			work[u11 + i__ + j * work_dim1] = work[cut + i__ + 
				invd * work_dim1] * work[u11 + i__ + j * 
				work_dim1] + work[cut + i__ + (invd + 1) * 
				work_dim1] * work[u11 + i__ + 1 + j * 
				work_dim1];
			work[u11 + i__ + 1 + j * work_dim1] = work[cut + i__ 
				+ 1 + invd * work_dim1] * u11_i_j__ + work[
				cut + i__ + 1 + (invd + 1) * work_dim1] * 
				u11_ip1_j__;
		    }
		    i__ += 2;
		}
	    }

/*       U11**T*invD1*U11->U11 */

	    i__1 = *n + *nb + 1;
	    qtrmm_("L", "U", "T", "U", &nnb, &nnb, &c_b11, &a[cut + 1 + (cut 
		    + 1) * a_dim1], lda, &work[u11 + 1 + work_dim1], &i__1);

	    i__1 = nnb;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = nnb;
		for (j = i__; j <= i__2; ++j) {
		    a[cut + i__ + (cut + j) * a_dim1] = work[u11 + i__ + j * 
			    work_dim1];
		}
	    }

/*          U01**T*invD*U01->A(CUT+I,CUT+J) */

	    i__1 = *n + *nb + 1;
	    i__2 = *n + *nb + 1;
	    qgemm_("T", "N", &nnb, &nnb, &cut, &c_b11, &a[(cut + 1) * a_dim1 
		    + 1], lda, &work[work_offset], &i__1, &c_b15, &work[u11 + 
		    1 + work_dim1], &i__2);

/*        U11 =  U11**T*invD1*U11 + U01**T*invD*U01 */

	    i__1 = nnb;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = nnb;
		for (j = i__; j <= i__2; ++j) {
		    a[cut + i__ + (cut + j) * a_dim1] += work[u11 + i__ + j * 
			    work_dim1];
		}
	    }

/*        U01 =  U00**T*invD0*U01 */

	    i__1 = *n + *nb + 1;
	    qtrmm_("L", uplo, "T", "U", &cut, &nnb, &c_b11, &a[a_offset], lda,
		     &work[work_offset], &i__1);

/*        Update U01 */

	    i__1 = cut;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = nnb;
		for (j = 1; j <= i__2; ++j) {
		    a[i__ + (cut + j) * a_dim1] = work[i__ + j * work_dim1];
		}
	    }

/*      Next Block */

	}

/*        Apply PERMUTATIONS P and P**T: P * inv(U**T)*inv(D)*inv(U) *P**T */

	i__ = 1;
	while(i__ <= *n) {
	    if (ipiv[i__] > 0) {
		ip = ipiv[i__];
		if (i__ < ip) {
		    qsyswapr_(uplo, n, &a[a_offset], lda, &i__, &ip);
		}
		if (i__ > ip) {
		    qsyswapr_(uplo, n, &a[a_offset], lda, &ip, &i__);
		}
	    } else {
		ip = -ipiv[i__];
		++i__;
		if (i__ - 1 < ip) {
		    i__1 = i__ - 1;
		    qsyswapr_(uplo, n, &a[a_offset], lda, &i__1, &ip);
		}
		if (i__ - 1 > ip) {
		    i__1 = i__ - 1;
		    qsyswapr_(uplo, n, &a[a_offset], lda, &ip, &i__1);
		}
	    }
	    ++i__;
	}
    } else {

/*        LOWER... */

/*        invA = P * inv(U**T)*inv(D)*inv(U)*P**T. */

	qtrtri_(uplo, "U", n, &a[a_offset], lda, info);

/*       inv(D) and inv(D)*inv(U) */

	k = *n;
	while(k >= 1) {
	    if (ipiv[k] > 0) {
/*           1 x 1 diagonal NNB */
		work[k + invd * work_dim1] = 1. / a[k + k * a_dim1];
		work[k + (invd + 1) * work_dim1] = 0.;
		--k;
	    } else {
/*           2 x 2 diagonal NNB */
		t = work[k - 1 + work_dim1];
		ak = a[k - 1 + (k - 1) * a_dim1] / t;
		akp1 = a[k + k * a_dim1] / t;
		akkp1 = work[k - 1 + work_dim1] / t;
		d__ = t * (ak * akp1 - 1.);
		work[k - 1 + invd * work_dim1] = akp1 / d__;
		work[k + invd * work_dim1] = ak / d__;
		work[k + (invd + 1) * work_dim1] = -akkp1 / d__;
		work[k - 1 + (invd + 1) * work_dim1] = -akkp1 / d__;
		k += -2;
	    }
	}

/*       inv(U**T) = (inv(U))**T */

/*       inv(U**T)*inv(D)*inv(U) */

	cut = 0;
	while(cut < *n) {
	    nnb = *nb;
	    if (cut + nnb > *n) {
		nnb = *n - cut;
	    } else {
		count = 0;
/*             count negative elements, */
		i__1 = cut + nnb;
		for (i__ = cut + 1; i__ <= i__1; ++i__) {
		    if (ipiv[i__] < 0) {
			++count;
		    }
		}
/*             need a even number for a clear cut */
		if (count % 2 == 1) {
		    ++nnb;
		}
	    }
/*     L21 Block */
	    i__1 = *n - cut - nnb;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = nnb;
		for (j = 1; j <= i__2; ++j) {
		    work[i__ + j * work_dim1] = a[cut + nnb + i__ + (cut + j) 
			    * a_dim1];
		}
	    }
/*     L11 Block */
	    i__1 = nnb;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		work[u11 + i__ + i__ * work_dim1] = 1.;
		i__2 = nnb;
		for (j = i__ + 1; j <= i__2; ++j) {
		    work[u11 + i__ + j * work_dim1] = 0.;
		}
		i__2 = i__ - 1;
		for (j = 1; j <= i__2; ++j) {
		    work[u11 + i__ + j * work_dim1] = a[cut + i__ + (cut + j) 
			    * a_dim1];
		}
	    }

/*          invD*L21 */

	    i__ = *n - cut - nnb;
	    while(i__ >= 1) {
		if (ipiv[cut + nnb + i__] > 0) {
		    i__1 = nnb;
		    for (j = 1; j <= i__1; ++j) {
			work[i__ + j * work_dim1] = work[cut + nnb + i__ + 
				invd * work_dim1] * work[i__ + j * work_dim1];
		    }
		    --i__;
		} else {
		    i__1 = nnb;
		    for (j = 1; j <= i__1; ++j) {
			u01_i_j__ = work[i__ + j * work_dim1];
			u01_ip1_j__ = work[i__ - 1 + j * work_dim1];
			work[i__ + j * work_dim1] = work[cut + nnb + i__ + 
				invd * work_dim1] * u01_i_j__ + work[cut + 
				nnb + i__ + (invd + 1) * work_dim1] * 
				u01_ip1_j__;
			work[i__ - 1 + j * work_dim1] = work[cut + nnb + i__ 
				- 1 + (invd + 1) * work_dim1] * u01_i_j__ + 
				work[cut + nnb + i__ - 1 + invd * work_dim1] *
				 u01_ip1_j__;
		    }
		    i__ += -2;
		}
	    }

/*        invD1*L11 */

	    i__ = nnb;
	    while(i__ >= 1) {
		if (ipiv[cut + i__] > 0) {
		    i__1 = nnb;
		    for (j = 1; j <= i__1; ++j) {
			work[u11 + i__ + j * work_dim1] = work[cut + i__ + 
				invd * work_dim1] * work[u11 + i__ + j * 
				work_dim1];
		    }
		    --i__;
		} else {
		    i__1 = nnb;
		    for (j = 1; j <= i__1; ++j) {
			u11_i_j__ = work[u11 + i__ + j * work_dim1];
			u11_ip1_j__ = work[u11 + i__ - 1 + j * work_dim1];
			work[u11 + i__ + j * work_dim1] = work[cut + i__ + 
				invd * work_dim1] * work[u11 + i__ + j * 
				work_dim1] + work[cut + i__ + (invd + 1) * 
				work_dim1] * u11_ip1_j__;
			work[u11 + i__ - 1 + j * work_dim1] = work[cut + i__ 
				- 1 + (invd + 1) * work_dim1] * u11_i_j__ + 
				work[cut + i__ - 1 + invd * work_dim1] * 
				u11_ip1_j__;
		    }
		    i__ += -2;
		}
	    }

/*       L11**T*invD1*L11->L11 */

	    i__1 = *n + *nb + 1;
	    qtrmm_("L", uplo, "T", "U", &nnb, &nnb, &c_b11, &a[cut + 1 + (cut 
		    + 1) * a_dim1], lda, &work[u11 + 1 + work_dim1], &i__1);

	    i__1 = nnb;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    a[cut + i__ + (cut + j) * a_dim1] = work[u11 + i__ + j * 
			    work_dim1];
		}
	    }

	    if (cut + nnb < *n) {

/*          L21**T*invD2*L21->A(CUT+I,CUT+J) */

		i__1 = *n - nnb - cut;
		i__2 = *n + *nb + 1;
		i__3 = *n + *nb + 1;
		qgemm_("T", "N", &nnb, &nnb, &i__1, &c_b11, &a[cut + nnb + 1 
			+ (cut + 1) * a_dim1], lda, &work[work_offset], &i__2,
			 &c_b15, &work[u11 + 1 + work_dim1], &i__3);

/*        L11 =  L11**T*invD1*L11 + U01**T*invD*U01 */

		i__1 = nnb;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = i__;
		    for (j = 1; j <= i__2; ++j) {
			a[cut + i__ + (cut + j) * a_dim1] += work[u11 + i__ + 
				j * work_dim1];
		    }
		}

/*        L01 =  L22**T*invD2*L21 */

		i__1 = *n - nnb - cut;
		i__2 = *n + *nb + 1;
		qtrmm_("L", uplo, "T", "U", &i__1, &nnb, &c_b11, &a[cut + nnb 
			+ 1 + (cut + nnb + 1) * a_dim1], lda, &work[
			work_offset], &i__2);

/*      Update L21 */

		i__1 = *n - cut - nnb;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = nnb;
		    for (j = 1; j <= i__2; ++j) {
			a[cut + nnb + i__ + (cut + j) * a_dim1] = work[i__ + 
				j * work_dim1];
		    }
		}
	    } else {

/*        L11 =  L11**T*invD1*L11 */

		i__1 = nnb;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = i__;
		    for (j = 1; j <= i__2; ++j) {
			a[cut + i__ + (cut + j) * a_dim1] = work[u11 + i__ + 
				j * work_dim1];
		    }
		}
	    }

/*      Next Block */

	    cut += nnb;
	}

/*        Apply PERMUTATIONS P and P**T: P * inv(U**T)*inv(D)*inv(U) *P**T */

	i__ = *n;
	while(i__ >= 1) {
	    if (ipiv[i__] > 0) {
		ip = ipiv[i__];
		if (i__ < ip) {
		    qsyswapr_(uplo, n, &a[a_offset], lda, &i__, &ip);
		}
		if (i__ > ip) {
		    qsyswapr_(uplo, n, &a[a_offset], lda, &ip, &i__);
		}
	    } else {
		ip = -ipiv[i__];
		if (i__ < ip) {
		    qsyswapr_(uplo, n, &a[a_offset], lda, &i__, &ip);
		}
		if (i__ > ip) {
		    qsyswapr_(uplo, n, &a[a_offset], lda, &ip, &i__);
		}
		--i__;
	    }
	    --i__;
	}
    }

    return;

/*     End of DSYTRI2X */

} /* qsytri2x_ */

