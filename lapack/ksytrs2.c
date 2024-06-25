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

#define __LAPACK_PRECISION_HALF
#include "f2c.h"

/* Table of constant values */

static halfcomplex c_b1 = {1.,0.};

/* > \brief \b ZSYTRS2 */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZSYTRS2 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zsytrs2
.f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zsytrs2
.f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zsytrs2
.f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZSYTRS2( UPLO, N, NRHS, A, LDA, IPIV, B, LDB, */
/*                           WORK, INFO ) */

/*       CHARACTER          UPLO */
/*       INTEGER            INFO, LDA, LDB, N, NRHS */
/*       INTEGER            IPIV( * ) */
/*       COMPLEX*16       A( LDA, * ), B( LDB, * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZSYTRS2 solves a system of linear equations A*X = B with a doublereal */
/* > symmetric matrix A using the factorization A = U*D*U**T or */
/* > A = L*D*L**T computed by ZSYTRF and converted by ZSYCONV. */
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
/* > \param[in] NRHS */
/* > \verbatim */
/* >          NRHS is INTEGER */
/* >          The number of right hand sides, i.e., the number of columns */
/* >          of the matrix B.  NRHS >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is COMPLEX*16 array, dimension (LDA,N) */
/* >          The block diagonal matrix D and the multipliers used to */
/* >          obtain the factor U or L as computed by ZSYTRF. */
/* >          Note that A is input / output. This might be counter-intuitive, */
/* >          and one may think that A is input only. A is input / output. This */
/* >          is because, at the start of the subroutine, we permute A in a */
/* >          "better" form and then we permute A back to its original form at */
/* >          the end. */
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
/* >          Details of the interchanges and the block structure of D */
/* >          as determined by ZSYTRF. */
/* > \endverbatim */
/* > */
/* > \param[in,out] B */
/* > \verbatim */
/* >          B is COMPLEX*16 array, dimension (LDB,NRHS) */
/* >          On entry, the right hand side matrix B. */
/* >          On exit, the solution matrix X. */
/* > \endverbatim */
/* > */
/* > \param[in] LDB */
/* > \verbatim */
/* >          LDB is INTEGER */
/* >          The leading dimension of the array B.  LDB >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is COMPLEX*16 array, dimension (N) */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date June 2016 */

/* > \ingroup complex16SYcomputational */

/*  ===================================================================== */
void  ksytrs2_(char *uplo, integer *n, integer *nrhs, 
	halfcomplex *a, integer *lda, integer *ipiv, halfcomplex *b, 
	integer *ldb, halfcomplex *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2;
    halfcomplex z__1, z__2, z__3;

    /* Local variables */
    integer i__, j, k;
    halfcomplex ak, bk;
    integer kp;
    halfcomplex akm1, bkm1, akm1k;
    extern logical lsame_(char *, char *);
    halfcomplex denom;
    integer iinfo;
    extern void  kscal_(integer *, halfcomplex *, 
	    halfcomplex *, integer *);
    logical upper;
    extern void  kswap_(integer *, halfcomplex *, integer *, 
	    halfcomplex *, integer *), ktrsm_(char *, char *, char *, char *
	    , integer *, integer *, halfcomplex *, halfcomplex *, integer 
	    *, halfcomplex *, integer *), 
	    xerbla_(char *, integer *), ksyconv_(char *, char *, 
	    integer *, halfcomplex *, integer *, integer *, halfcomplex *,
	     integer *);


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     June 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --work;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*nrhs < 0) {
	*info = -3;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -5;
    } else if (*ldb < f2cmax(1,*n)) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZSYTRS2", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*n == 0 || *nrhs == 0) {
	return;
    }

/*     Convert A */

    ksyconv_(uplo, "C", n, &a[a_offset], lda, &ipiv[1], &work[1], &iinfo);

    if (upper) {

/*        Solve A*X = B, where A = U*D*U**T. */

/*       P**T * B */
	k = *n;
	while(k >= 1) {
	    if (ipiv[k] > 0) {
/*           1 x 1 diagonal block */
/*           Interchange rows K and IPIV(K). */
		kp = ipiv[k];
		if (kp != k) {
		    kswap_(nrhs, &b[k + b_dim1], ldb, &b[kp + b_dim1], ldb);
		}
		--k;
	    } else {
/*           2 x 2 diagonal block */
/*           Interchange rows K-1 and -IPIV(K). */
		kp = -ipiv[k];
		if (kp == -ipiv[k - 1]) {
		    kswap_(nrhs, &b[k - 1 + b_dim1], ldb, &b[kp + b_dim1], 
			    ldb);
		}
		k += -2;
	    }
	}

/*  Compute (U \P**T * B) -> B    [ (U \P**T * B) ] */

	ktrsm_("L", "U", "N", "U", n, nrhs, &c_b1, &a[a_offset], lda, &b[
		b_offset], ldb);

/*  Compute D \ B -> B   [ D \ (U \P**T * B) ] */

	i__ = *n;
	while(i__ >= 1) {
	    if (ipiv[i__] > 0) {
		z_div(&z__1, &c_b1, &a[i__ + i__ * a_dim1]);
		kscal_(nrhs, &z__1, &b[i__ + b_dim1], ldb);
	    } else if (i__ > 1) {
		if (ipiv[i__ - 1] == ipiv[i__]) {
		    i__1 = i__;
		    akm1k.r = work[i__1].r, akm1k.i = work[i__1].i;
		    z_div(&z__1, &a[i__ - 1 + (i__ - 1) * a_dim1], &akm1k);
		    akm1.r = z__1.r, akm1.i = z__1.i;
		    z_div(&z__1, &a[i__ + i__ * a_dim1], &akm1k);
		    ak.r = z__1.r, ak.i = z__1.i;
		    z__2.r = akm1.r * ak.r - akm1.i * ak.i, z__2.i = akm1.r * 
			    ak.i + akm1.i * ak.r;
		    z__1.r = z__2.r - 1., z__1.i = z__2.i - 0.;
		    denom.r = z__1.r, denom.i = z__1.i;
		    i__1 = *nrhs;
		    for (j = 1; j <= i__1; ++j) {
			z_div(&z__1, &b[i__ - 1 + j * b_dim1], &akm1k);
			bkm1.r = z__1.r, bkm1.i = z__1.i;
			z_div(&z__1, &b[i__ + j * b_dim1], &akm1k);
			bk.r = z__1.r, bk.i = z__1.i;
			i__2 = i__ - 1 + j * b_dim1;
			z__3.r = ak.r * bkm1.r - ak.i * bkm1.i, z__3.i = ak.r 
				* bkm1.i + ak.i * bkm1.r;
			z__2.r = z__3.r - bk.r, z__2.i = z__3.i - bk.i;
			z_div(&z__1, &z__2, &denom);
			b[i__2].r = z__1.r, b[i__2].i = z__1.i;
			i__2 = i__ + j * b_dim1;
			z__3.r = akm1.r * bk.r - akm1.i * bk.i, z__3.i = 
				akm1.r * bk.i + akm1.i * bk.r;
			z__2.r = z__3.r - bkm1.r, z__2.i = z__3.i - bkm1.i;
			z_div(&z__1, &z__2, &denom);
			b[i__2].r = z__1.r, b[i__2].i = z__1.i;
/* L15: */
		    }
		    --i__;
		}
	    }
	    --i__;
	}

/*      Compute (U**T \ B) -> B   [ U**T \ (D \ (U \P**T * B) ) ] */

	ktrsm_("L", "U", "T", "U", n, nrhs, &c_b1, &a[a_offset], lda, &b[
		b_offset], ldb);

/*       P * B  [ P * (U**T \ (D \ (U \P**T * B) )) ] */

	k = 1;
	while(k <= *n) {
	    if (ipiv[k] > 0) {
/*           1 x 1 diagonal block */
/*           Interchange rows K and IPIV(K). */
		kp = ipiv[k];
		if (kp != k) {
		    kswap_(nrhs, &b[k + b_dim1], ldb, &b[kp + b_dim1], ldb);
		}
		++k;
	    } else {
/*           2 x 2 diagonal block */
/*           Interchange rows K-1 and -IPIV(K). */
		kp = -ipiv[k];
		if (k < *n && kp == -ipiv[k + 1]) {
		    kswap_(nrhs, &b[k + b_dim1], ldb, &b[kp + b_dim1], ldb);
		}
		k += 2;
	    }
	}

    } else {

/*        Solve A*X = B, where A = L*D*L**T. */

/*       P**T * B */
	k = 1;
	while(k <= *n) {
	    if (ipiv[k] > 0) {
/*           1 x 1 diagonal block */
/*           Interchange rows K and IPIV(K). */
		kp = ipiv[k];
		if (kp != k) {
		    kswap_(nrhs, &b[k + b_dim1], ldb, &b[kp + b_dim1], ldb);
		}
		++k;
	    } else {
/*           2 x 2 diagonal block */
/*           Interchange rows K and -IPIV(K+1). */
		kp = -ipiv[k + 1];
		if (kp == -ipiv[k]) {
		    kswap_(nrhs, &b[k + 1 + b_dim1], ldb, &b[kp + b_dim1], 
			    ldb);
		}
		k += 2;
	    }
	}

/*  Compute (L \P**T * B) -> B    [ (L \P**T * B) ] */

	ktrsm_("L", "L", "N", "U", n, nrhs, &c_b1, &a[a_offset], lda, &b[
		b_offset], ldb);

/*  Compute D \ B -> B   [ D \ (L \P**T * B) ] */

	i__ = 1;
	while(i__ <= *n) {
	    if (ipiv[i__] > 0) {
		z_div(&z__1, &c_b1, &a[i__ + i__ * a_dim1]);
		kscal_(nrhs, &z__1, &b[i__ + b_dim1], ldb);
	    } else {
		i__1 = i__;
		akm1k.r = work[i__1].r, akm1k.i = work[i__1].i;
		z_div(&z__1, &a[i__ + i__ * a_dim1], &akm1k);
		akm1.r = z__1.r, akm1.i = z__1.i;
		z_div(&z__1, &a[i__ + 1 + (i__ + 1) * a_dim1], &akm1k);
		ak.r = z__1.r, ak.i = z__1.i;
		z__2.r = akm1.r * ak.r - akm1.i * ak.i, z__2.i = akm1.r * 
			ak.i + akm1.i * ak.r;
		z__1.r = z__2.r - 1., z__1.i = z__2.i - 0.;
		denom.r = z__1.r, denom.i = z__1.i;
		i__1 = *nrhs;
		for (j = 1; j <= i__1; ++j) {
		    z_div(&z__1, &b[i__ + j * b_dim1], &akm1k);
		    bkm1.r = z__1.r, bkm1.i = z__1.i;
		    z_div(&z__1, &b[i__ + 1 + j * b_dim1], &akm1k);
		    bk.r = z__1.r, bk.i = z__1.i;
		    i__2 = i__ + j * b_dim1;
		    z__3.r = ak.r * bkm1.r - ak.i * bkm1.i, z__3.i = ak.r * 
			    bkm1.i + ak.i * bkm1.r;
		    z__2.r = z__3.r - bk.r, z__2.i = z__3.i - bk.i;
		    z_div(&z__1, &z__2, &denom);
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		    i__2 = i__ + 1 + j * b_dim1;
		    z__3.r = akm1.r * bk.r - akm1.i * bk.i, z__3.i = akm1.r * 
			    bk.i + akm1.i * bk.r;
		    z__2.r = z__3.r - bkm1.r, z__2.i = z__3.i - bkm1.i;
		    z_div(&z__1, &z__2, &denom);
		    b[i__2].r = z__1.r, b[i__2].i = z__1.i;
/* L25: */
		}
		++i__;
	    }
	    ++i__;
	}

/*  Compute (L**T \ B) -> B   [ L**T \ (D \ (L \P**T * B) ) ] */

	ktrsm_("L", "L", "T", "U", n, nrhs, &c_b1, &a[a_offset], lda, &b[
		b_offset], ldb);

/*       P * B  [ P * (L**T \ (D \ (L \P**T * B) )) ] */

	k = *n;
	while(k >= 1) {
	    if (ipiv[k] > 0) {
/*           1 x 1 diagonal block */
/*           Interchange rows K and IPIV(K). */
		kp = ipiv[k];
		if (kp != k) {
		    kswap_(nrhs, &b[k + b_dim1], ldb, &b[kp + b_dim1], ldb);
		}
		--k;
	    } else {
/*           2 x 2 diagonal block */
/*           Interchange rows K-1 and -IPIV(K). */
		kp = -ipiv[k];
		if (k > 1 && kp == -ipiv[k - 1]) {
		    kswap_(nrhs, &b[k + b_dim1], ldb, &b[kp + b_dim1], ldb);
		}
		k += -2;
	    }
	}

    }

/*     Revert A */

    ksyconv_(uplo, "R", n, &a[a_offset], lda, &ipiv[1], &work[1], &iinfo);

    return;

/*     End of ZSYTRS2 */

} /* ksytrs2_ */

