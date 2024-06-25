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

#define __LAPACK_PRECISION_DOUBLE
#include "f2c.h"

/* Table of constant values */

static doublecomplex c_b1 = {1.,0.};
static integer c__1 = 1;

/* > \brief \b ZSPTRS */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZSPTRS + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zsptrs.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zsptrs.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zsptrs.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZSPTRS( UPLO, N, NRHS, AP, IPIV, B, LDB, INFO ) */

/*       CHARACTER          UPLO */
/*       INTEGER            INFO, LDB, N, NRHS */
/*       INTEGER            IPIV( * ) */
/*       COMPLEX*16         AP( * ), B( LDB, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZSPTRS solves a system of linear equations A*X = B with a complex */
/* > symmetric matrix A stored in packed format using the factorization */
/* > A = U*D*U**T or A = L*D*L**T computed by ZSPTRF. */
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
/* > \param[in] AP */
/* > \verbatim */
/* >          AP is COMPLEX*16 array, dimension (N*(N+1)/2) */
/* >          The block diagonal matrix D and the multipliers used to */
/* >          obtain the factor U or L as computed by ZSPTRF, stored as a */
/* >          packed triangular matrix. */
/* > \endverbatim */
/* > */
/* > \param[in] IPIV */
/* > \verbatim */
/* >          IPIV is INTEGER array, dimension (N) */
/* >          Details of the interchanges and the block structure of D */
/* >          as determined by ZSPTRF. */
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
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          < 0: if INFO = -i, the i-th argument had an illegal value */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complex16OTHERcomputational */

/*  ===================================================================== */
void  zsptrs_(char *uplo, integer *n, integer *nrhs, 
	doublecomplex *ap, integer *ipiv, doublecomplex *b, integer *ldb, 
	integer *info)
{
    /* System generated locals */
    integer b_dim1, b_offset, i__1, i__2;
    doublecomplex z__1, z__2, z__3;

    /* Local variables */
    integer j, k;
    doublecomplex ak, bk;
    integer kc, kp;
    doublecomplex akm1, bkm1, akm1k;
    extern logical lsame_(char *, char *);
    doublecomplex denom;
    extern void  zscal_(integer *, doublecomplex *, 
	    doublecomplex *, integer *), zgemv_(char *, integer *, integer *, 
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *, 
	    integer *, doublecomplex *, doublecomplex *, integer *);
    logical upper;
    extern void  zgeru_(integer *, integer *, doublecomplex *, 
	    doublecomplex *, integer *, doublecomplex *, integer *, 
	    doublecomplex *, integer *), zswap_(integer *, doublecomplex *, 
	    integer *, doublecomplex *, integer *), xerbla_(char *, integer *);


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    --ap;
    --ipiv;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*nrhs < 0) {
	*info = -3;
    } else if (*ldb < f2cmax(1,*n)) {
	*info = -7;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZSPTRS", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*n == 0 || *nrhs == 0) {
	return;
    }

    if (upper) {

/*        Solve A*X = B, where A = U*D*U**T. */

/*        First solve U*D*X = B, overwriting B with X. */

/*        K is the main loop index, decreasing from N to 1 in steps of */
/*        1 or 2, depending on the size of the diagonal blocks. */

	k = *n;
	kc = *n * (*n + 1) / 2 + 1;
L10:

/*        If K < 1, exit from loop. */

	if (k < 1) {
	    goto L30;
	}

	kc -= k;
	if (ipiv[k] > 0) {

/*           1 x 1 diagonal block */

/*           Interchange rows K and IPIV(K). */

	    kp = ipiv[k];
	    if (kp != k) {
		zswap_(nrhs, &b[k + b_dim1], ldb, &b[kp + b_dim1], ldb);
	    }

/*           Multiply by inv(U(K)), where U(K) is the transformation */
/*           stored in column K of A. */

	    i__1 = k - 1;
	    z__1.r = -1., z__1.i = -0.;
	    zgeru_(&i__1, nrhs, &z__1, &ap[kc], &c__1, &b[k + b_dim1], ldb, &
		    b[b_dim1 + 1], ldb);

/*           Multiply by the inverse of the diagonal block. */

	    z_div(&z__1, &c_b1, &ap[kc + k - 1]);
	    zscal_(nrhs, &z__1, &b[k + b_dim1], ldb);
	    --k;
	} else {

/*           2 x 2 diagonal block */

/*           Interchange rows K-1 and -IPIV(K). */

	    kp = -ipiv[k];
	    if (kp != k - 1) {
		zswap_(nrhs, &b[k - 1 + b_dim1], ldb, &b[kp + b_dim1], ldb);
	    }

/*           Multiply by inv(U(K)), where U(K) is the transformation */
/*           stored in columns K-1 and K of A. */

	    i__1 = k - 2;
	    z__1.r = -1., z__1.i = -0.;
	    zgeru_(&i__1, nrhs, &z__1, &ap[kc], &c__1, &b[k + b_dim1], ldb, &
		    b[b_dim1 + 1], ldb);
	    i__1 = k - 2;
	    z__1.r = -1., z__1.i = -0.;
	    zgeru_(&i__1, nrhs, &z__1, &ap[kc - (k - 1)], &c__1, &b[k - 1 + 
		    b_dim1], ldb, &b[b_dim1 + 1], ldb);

/*           Multiply by the inverse of the diagonal block. */

	    i__1 = kc + k - 2;
	    akm1k.r = ap[i__1].r, akm1k.i = ap[i__1].i;
	    z_div(&z__1, &ap[kc - 1], &akm1k);
	    akm1.r = z__1.r, akm1.i = z__1.i;
	    z_div(&z__1, &ap[kc + k - 1], &akm1k);
	    ak.r = z__1.r, ak.i = z__1.i;
	    z__2.r = akm1.r * ak.r - akm1.i * ak.i, z__2.i = akm1.r * ak.i + 
		    akm1.i * ak.r;
	    z__1.r = z__2.r - 1., z__1.i = z__2.i - 0.;
	    denom.r = z__1.r, denom.i = z__1.i;
	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {
		z_div(&z__1, &b[k - 1 + j * b_dim1], &akm1k);
		bkm1.r = z__1.r, bkm1.i = z__1.i;
		z_div(&z__1, &b[k + j * b_dim1], &akm1k);
		bk.r = z__1.r, bk.i = z__1.i;
		i__2 = k - 1 + j * b_dim1;
		z__3.r = ak.r * bkm1.r - ak.i * bkm1.i, z__3.i = ak.r * 
			bkm1.i + ak.i * bkm1.r;
		z__2.r = z__3.r - bk.r, z__2.i = z__3.i - bk.i;
		z_div(&z__1, &z__2, &denom);
		b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		i__2 = k + j * b_dim1;
		z__3.r = akm1.r * bk.r - akm1.i * bk.i, z__3.i = akm1.r * 
			bk.i + akm1.i * bk.r;
		z__2.r = z__3.r - bkm1.r, z__2.i = z__3.i - bkm1.i;
		z_div(&z__1, &z__2, &denom);
		b[i__2].r = z__1.r, b[i__2].i = z__1.i;
/* L20: */
	    }
	    kc = kc - k + 1;
	    k += -2;
	}

	goto L10;
L30:

/*        Next solve U**T*X = B, overwriting B with X. */

/*        K is the main loop index, increasing from 1 to N in steps of */
/*        1 or 2, depending on the size of the diagonal blocks. */

	k = 1;
	kc = 1;
L40:

/*        If K > N, exit from loop. */

	if (k > *n) {
	    goto L50;
	}

	if (ipiv[k] > 0) {

/*           1 x 1 diagonal block */

/*           Multiply by inv(U**T(K)), where U(K) is the transformation */
/*           stored in column K of A. */

	    i__1 = k - 1;
	    z__1.r = -1., z__1.i = -0.;
	    zgemv_("Transpose", &i__1, nrhs, &z__1, &b[b_offset], ldb, &ap[kc]
		    , &c__1, &c_b1, &b[k + b_dim1], ldb);

/*           Interchange rows K and IPIV(K). */

	    kp = ipiv[k];
	    if (kp != k) {
		zswap_(nrhs, &b[k + b_dim1], ldb, &b[kp + b_dim1], ldb);
	    }
	    kc += k;
	    ++k;
	} else {

/*           2 x 2 diagonal block */

/*           Multiply by inv(U**T(K+1)), where U(K+1) is the transformation */
/*           stored in columns K and K+1 of A. */

	    i__1 = k - 1;
	    z__1.r = -1., z__1.i = -0.;
	    zgemv_("Transpose", &i__1, nrhs, &z__1, &b[b_offset], ldb, &ap[kc]
		    , &c__1, &c_b1, &b[k + b_dim1], ldb);
	    i__1 = k - 1;
	    z__1.r = -1., z__1.i = -0.;
	    zgemv_("Transpose", &i__1, nrhs, &z__1, &b[b_offset], ldb, &ap[kc 
		    + k], &c__1, &c_b1, &b[k + 1 + b_dim1], ldb);

/*           Interchange rows K and -IPIV(K). */

	    kp = -ipiv[k];
	    if (kp != k) {
		zswap_(nrhs, &b[k + b_dim1], ldb, &b[kp + b_dim1], ldb);
	    }
	    kc = kc + (k << 1) + 1;
	    k += 2;
	}

	goto L40;
L50:

	;
    } else {

/*        Solve A*X = B, where A = L*D*L**T. */

/*        First solve L*D*X = B, overwriting B with X. */

/*        K is the main loop index, increasing from 1 to N in steps of */
/*        1 or 2, depending on the size of the diagonal blocks. */

	k = 1;
	kc = 1;
L60:

/*        If K > N, exit from loop. */

	if (k > *n) {
	    goto L80;
	}

	if (ipiv[k] > 0) {

/*           1 x 1 diagonal block */

/*           Interchange rows K and IPIV(K). */

	    kp = ipiv[k];
	    if (kp != k) {
		zswap_(nrhs, &b[k + b_dim1], ldb, &b[kp + b_dim1], ldb);
	    }

/*           Multiply by inv(L(K)), where L(K) is the transformation */
/*           stored in column K of A. */

	    if (k < *n) {
		i__1 = *n - k;
		z__1.r = -1., z__1.i = -0.;
		zgeru_(&i__1, nrhs, &z__1, &ap[kc + 1], &c__1, &b[k + b_dim1],
			 ldb, &b[k + 1 + b_dim1], ldb);
	    }

/*           Multiply by the inverse of the diagonal block. */

	    z_div(&z__1, &c_b1, &ap[kc]);
	    zscal_(nrhs, &z__1, &b[k + b_dim1], ldb);
	    kc = kc + *n - k + 1;
	    ++k;
	} else {

/*           2 x 2 diagonal block */

/*           Interchange rows K+1 and -IPIV(K). */

	    kp = -ipiv[k];
	    if (kp != k + 1) {
		zswap_(nrhs, &b[k + 1 + b_dim1], ldb, &b[kp + b_dim1], ldb);
	    }

/*           Multiply by inv(L(K)), where L(K) is the transformation */
/*           stored in columns K and K+1 of A. */

	    if (k < *n - 1) {
		i__1 = *n - k - 1;
		z__1.r = -1., z__1.i = -0.;
		zgeru_(&i__1, nrhs, &z__1, &ap[kc + 2], &c__1, &b[k + b_dim1],
			 ldb, &b[k + 2 + b_dim1], ldb);
		i__1 = *n - k - 1;
		z__1.r = -1., z__1.i = -0.;
		zgeru_(&i__1, nrhs, &z__1, &ap[kc + *n - k + 2], &c__1, &b[k 
			+ 1 + b_dim1], ldb, &b[k + 2 + b_dim1], ldb);
	    }

/*           Multiply by the inverse of the diagonal block. */

	    i__1 = kc + 1;
	    akm1k.r = ap[i__1].r, akm1k.i = ap[i__1].i;
	    z_div(&z__1, &ap[kc], &akm1k);
	    akm1.r = z__1.r, akm1.i = z__1.i;
	    z_div(&z__1, &ap[kc + *n - k + 1], &akm1k);
	    ak.r = z__1.r, ak.i = z__1.i;
	    z__2.r = akm1.r * ak.r - akm1.i * ak.i, z__2.i = akm1.r * ak.i + 
		    akm1.i * ak.r;
	    z__1.r = z__2.r - 1., z__1.i = z__2.i - 0.;
	    denom.r = z__1.r, denom.i = z__1.i;
	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {
		z_div(&z__1, &b[k + j * b_dim1], &akm1k);
		bkm1.r = z__1.r, bkm1.i = z__1.i;
		z_div(&z__1, &b[k + 1 + j * b_dim1], &akm1k);
		bk.r = z__1.r, bk.i = z__1.i;
		i__2 = k + j * b_dim1;
		z__3.r = ak.r * bkm1.r - ak.i * bkm1.i, z__3.i = ak.r * 
			bkm1.i + ak.i * bkm1.r;
		z__2.r = z__3.r - bk.r, z__2.i = z__3.i - bk.i;
		z_div(&z__1, &z__2, &denom);
		b[i__2].r = z__1.r, b[i__2].i = z__1.i;
		i__2 = k + 1 + j * b_dim1;
		z__3.r = akm1.r * bk.r - akm1.i * bk.i, z__3.i = akm1.r * 
			bk.i + akm1.i * bk.r;
		z__2.r = z__3.r - bkm1.r, z__2.i = z__3.i - bkm1.i;
		z_div(&z__1, &z__2, &denom);
		b[i__2].r = z__1.r, b[i__2].i = z__1.i;
/* L70: */
	    }
	    kc = kc + (*n - k << 1) + 1;
	    k += 2;
	}

	goto L60;
L80:

/*        Next solve L**T*X = B, overwriting B with X. */

/*        K is the main loop index, decreasing from N to 1 in steps of */
/*        1 or 2, depending on the size of the diagonal blocks. */

	k = *n;
	kc = *n * (*n + 1) / 2 + 1;
L90:

/*        If K < 1, exit from loop. */

	if (k < 1) {
	    goto L100;
	}

	kc -= *n - k + 1;
	if (ipiv[k] > 0) {

/*           1 x 1 diagonal block */

/*           Multiply by inv(L**T(K)), where L(K) is the transformation */
/*           stored in column K of A. */

	    if (k < *n) {
		i__1 = *n - k;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("Transpose", &i__1, nrhs, &z__1, &b[k + 1 + b_dim1], 
			ldb, &ap[kc + 1], &c__1, &c_b1, &b[k + b_dim1], ldb);
	    }

/*           Interchange rows K and IPIV(K). */

	    kp = ipiv[k];
	    if (kp != k) {
		zswap_(nrhs, &b[k + b_dim1], ldb, &b[kp + b_dim1], ldb);
	    }
	    --k;
	} else {

/*           2 x 2 diagonal block */

/*           Multiply by inv(L**T(K-1)), where L(K-1) is the transformation */
/*           stored in columns K-1 and K of A. */

	    if (k < *n) {
		i__1 = *n - k;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("Transpose", &i__1, nrhs, &z__1, &b[k + 1 + b_dim1], 
			ldb, &ap[kc + 1], &c__1, &c_b1, &b[k + b_dim1], ldb);
		i__1 = *n - k;
		z__1.r = -1., z__1.i = -0.;
		zgemv_("Transpose", &i__1, nrhs, &z__1, &b[k + 1 + b_dim1], 
			ldb, &ap[kc - (*n - k)], &c__1, &c_b1, &b[k - 1 + 
			b_dim1], ldb);
	    }

/*           Interchange rows K and -IPIV(K). */

	    kp = -ipiv[k];
	    if (kp != k) {
		zswap_(nrhs, &b[k + b_dim1], ldb, &b[kp + b_dim1], ldb);
	    }
	    kc -= *n - k + 2;
	    k += -2;
	}

	goto L90;
L100:
	;
    }

    return;

/*     End of ZSPTRS */

} /* zsptrs_ */

