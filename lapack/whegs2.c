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

static quadcomplex c_b1 = {1.,0.};
static integer c__1 = 1;

/* > \brief \b ZHEGS2 reduces a Hermitian definite generalized eigenproblem to standard form, using the factor
ization results obtained from cpotrf (unblocked algorithm). */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZHEGS2 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zhegs2.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zhegs2.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zhegs2.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZHEGS2( ITYPE, UPLO, N, A, LDA, B, LDB, INFO ) */

/*       CHARACTER          UPLO */
/*       INTEGER            INFO, ITYPE, LDA, LDB, N */
/*       COMPLEX*16         A( LDA, * ), B( LDB, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZHEGS2 reduces a complex Hermitian-definite generalized */
/* > eigenproblem to standard form. */
/* > */
/* > If ITYPE = 1, the problem is A*x = lambda*B*x, */
/* > and A is overwritten by inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H) */
/* > */
/* > If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or */
/* > B*A*x = lambda*x, and A is overwritten by U*A*U**H or L**H *A*L. */
/* > */
/* > B must have been previously factorized as U**H *U or L*L**H by ZPOTRF. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] ITYPE */
/* > \verbatim */
/* >          ITYPE is INTEGER */
/* >          = 1: compute inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H); */
/* >          = 2 or 3: compute U*A*U**H or L**H *A*L. */
/* > \endverbatim */
/* > */
/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          Specifies whether the upper or lower triangular part of the */
/* >          Hermitian matrix A is stored, and how B has been factorized. */
/* >          = 'U':  Upper triangular */
/* >          = 'L':  Lower triangular */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrices A and B.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is COMPLEX*16 array, dimension (LDA,N) */
/* >          On entry, the Hermitian matrix A.  If UPLO = 'U', the leading */
/* >          n by n upper triangular part of A contains the upper */
/* >          triangular part of the matrix A, and the strictly lower */
/* >          triangular part of A is not referenced.  If UPLO = 'L', the */
/* >          leading n by n lower triangular part of A contains the lower */
/* >          triangular part of the matrix A, and the strictly upper */
/* >          triangular part of A is not referenced. */
/* > */
/* >          On exit, if INFO = 0, the transformed matrix, stored in the */
/* >          same format as A. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in,out] B */
/* > \verbatim */
/* >          B is COMPLEX*16 array, dimension (LDB,N) */
/* >          The triangular factor from the Cholesky factorization of B, */
/* >          as returned by ZPOTRF. */
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
/* >          = 0:  successful exit. */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complex16HEcomputational */

/*  ===================================================================== */
void  whegs2_(integer *itype, char *uplo, integer *n, 
	quadcomplex *a, integer *lda, quadcomplex *b, integer *ldb, 
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2;
    quadreal d__1, d__2;
    quadcomplex z__1;

    /* Local variables */
    integer k;
    quadcomplex ct;
    quadreal akk, bkk;
    extern void  wher2_(char *, integer *, quadcomplex *, 
	    quadcomplex *, integer *, quadcomplex *, integer *, 
	    quadcomplex *, integer *);
    extern logical lsame_(char *, char *);
    logical upper;
    extern void  waxpy_(integer *, quadcomplex *, 
	    quadcomplex *, integer *, quadcomplex *, integer *), wtrmv_(
	    char *, char *, char *, integer *, quadcomplex *, integer *, 
	    quadcomplex *, integer *), wtrsv_(char *
	    , char *, char *, integer *, quadcomplex *, integer *, 
	    quadcomplex *, integer *), xerbla_(char 
	    *, integer *), wqscal_(integer *, quadreal *, 
	    quadcomplex *, integer *), wlacgv_(integer *, quadcomplex *, 
	    integer *);


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (*itype < 1 || *itype > 3) {
	*info = -1;
    } else if (! upper && ! lsame_(uplo, "L")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -5;
    } else if (*ldb < f2cmax(1,*n)) {
	*info = -7;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZHEGS2", &i__1);
	return;
    }

    if (*itype == 1) {
	if (upper) {

/*           Compute inv(U**H)*A*inv(U) */

	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {

/*              Update the upper triangle of A(k:n,k:n) */

		i__2 = k + k * a_dim1;
		akk = a[i__2].r;
		i__2 = k + k * b_dim1;
		bkk = b[i__2].r;
/* Computing 2nd power */
		d__1 = bkk;
		akk /= d__1 * d__1;
		i__2 = k + k * a_dim1;
		a[i__2].r = akk, a[i__2].i = 0.;
		if (k < *n) {
		    i__2 = *n - k;
		    d__1 = 1. / bkk;
		    wqscal_(&i__2, &d__1, &a[k + (k + 1) * a_dim1], lda);
		    d__1 = akk * -.5;
		    ct.r = d__1, ct.i = 0.;
		    i__2 = *n - k;
		    wlacgv_(&i__2, &a[k + (k + 1) * a_dim1], lda);
		    i__2 = *n - k;
		    wlacgv_(&i__2, &b[k + (k + 1) * b_dim1], ldb);
		    i__2 = *n - k;
		    waxpy_(&i__2, &ct, &b[k + (k + 1) * b_dim1], ldb, &a[k + (
			    k + 1) * a_dim1], lda);
		    i__2 = *n - k;
		    z__1.r = -1., z__1.i = -0.;
		    wher2_(uplo, &i__2, &z__1, &a[k + (k + 1) * a_dim1], lda, 
			    &b[k + (k + 1) * b_dim1], ldb, &a[k + 1 + (k + 1) 
			    * a_dim1], lda);
		    i__2 = *n - k;
		    waxpy_(&i__2, &ct, &b[k + (k + 1) * b_dim1], ldb, &a[k + (
			    k + 1) * a_dim1], lda);
		    i__2 = *n - k;
		    wlacgv_(&i__2, &b[k + (k + 1) * b_dim1], ldb);
		    i__2 = *n - k;
		    wtrsv_(uplo, "Conjugate transpose", "Non-unit", &i__2, &b[
			    k + 1 + (k + 1) * b_dim1], ldb, &a[k + (k + 1) * 
			    a_dim1], lda);
		    i__2 = *n - k;
		    wlacgv_(&i__2, &a[k + (k + 1) * a_dim1], lda);
		}
/* L10: */
	    }
	} else {

/*           Compute inv(L)*A*inv(L**H) */

	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {

/*              Update the lower triangle of A(k:n,k:n) */

		i__2 = k + k * a_dim1;
		akk = a[i__2].r;
		i__2 = k + k * b_dim1;
		bkk = b[i__2].r;
/* Computing 2nd power */
		d__1 = bkk;
		akk /= d__1 * d__1;
		i__2 = k + k * a_dim1;
		a[i__2].r = akk, a[i__2].i = 0.;
		if (k < *n) {
		    i__2 = *n - k;
		    d__1 = 1. / bkk;
		    wqscal_(&i__2, &d__1, &a[k + 1 + k * a_dim1], &c__1);
		    d__1 = akk * -.5;
		    ct.r = d__1, ct.i = 0.;
		    i__2 = *n - k;
		    waxpy_(&i__2, &ct, &b[k + 1 + k * b_dim1], &c__1, &a[k + 
			    1 + k * a_dim1], &c__1);
		    i__2 = *n - k;
		    z__1.r = -1., z__1.i = -0.;
		    wher2_(uplo, &i__2, &z__1, &a[k + 1 + k * a_dim1], &c__1, 
			    &b[k + 1 + k * b_dim1], &c__1, &a[k + 1 + (k + 1) 
			    * a_dim1], lda);
		    i__2 = *n - k;
		    waxpy_(&i__2, &ct, &b[k + 1 + k * b_dim1], &c__1, &a[k + 
			    1 + k * a_dim1], &c__1);
		    i__2 = *n - k;
		    wtrsv_(uplo, "No transpose", "Non-unit", &i__2, &b[k + 1 
			    + (k + 1) * b_dim1], ldb, &a[k + 1 + k * a_dim1], 
			    &c__1);
		}
/* L20: */
	    }
	}
    } else {
	if (upper) {

/*           Compute U*A*U**H */

	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {

/*              Update the upper triangle of A(1:k,1:k) */

		i__2 = k + k * a_dim1;
		akk = a[i__2].r;
		i__2 = k + k * b_dim1;
		bkk = b[i__2].r;
		i__2 = k - 1;
		wtrmv_(uplo, "No transpose", "Non-unit", &i__2, &b[b_offset], 
			ldb, &a[k * a_dim1 + 1], &c__1);
		d__1 = akk * .5;
		ct.r = d__1, ct.i = 0.;
		i__2 = k - 1;
		waxpy_(&i__2, &ct, &b[k * b_dim1 + 1], &c__1, &a[k * a_dim1 + 
			1], &c__1);
		i__2 = k - 1;
		wher2_(uplo, &i__2, &c_b1, &a[k * a_dim1 + 1], &c__1, &b[k * 
			b_dim1 + 1], &c__1, &a[a_offset], lda);
		i__2 = k - 1;
		waxpy_(&i__2, &ct, &b[k * b_dim1 + 1], &c__1, &a[k * a_dim1 + 
			1], &c__1);
		i__2 = k - 1;
		wqscal_(&i__2, &bkk, &a[k * a_dim1 + 1], &c__1);
		i__2 = k + k * a_dim1;
/* Computing 2nd power */
		d__2 = bkk;
		d__1 = akk * (d__2 * d__2);
		a[i__2].r = d__1, a[i__2].i = 0.;
/* L30: */
	    }
	} else {

/*           Compute L**H *A*L */

	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {

/*              Update the lower triangle of A(1:k,1:k) */

		i__2 = k + k * a_dim1;
		akk = a[i__2].r;
		i__2 = k + k * b_dim1;
		bkk = b[i__2].r;
		i__2 = k - 1;
		wlacgv_(&i__2, &a[k + a_dim1], lda);
		i__2 = k - 1;
		wtrmv_(uplo, "Conjugate transpose", "Non-unit", &i__2, &b[
			b_offset], ldb, &a[k + a_dim1], lda);
		d__1 = akk * .5;
		ct.r = d__1, ct.i = 0.;
		i__2 = k - 1;
		wlacgv_(&i__2, &b[k + b_dim1], ldb);
		i__2 = k - 1;
		waxpy_(&i__2, &ct, &b[k + b_dim1], ldb, &a[k + a_dim1], lda);
		i__2 = k - 1;
		wher2_(uplo, &i__2, &c_b1, &a[k + a_dim1], lda, &b[k + b_dim1]
			, ldb, &a[a_offset], lda);
		i__2 = k - 1;
		waxpy_(&i__2, &ct, &b[k + b_dim1], ldb, &a[k + a_dim1], lda);
		i__2 = k - 1;
		wlacgv_(&i__2, &b[k + b_dim1], ldb);
		i__2 = k - 1;
		wqscal_(&i__2, &bkk, &a[k + a_dim1], lda);
		i__2 = k - 1;
		wlacgv_(&i__2, &a[k + a_dim1], lda);
		i__2 = k + k * a_dim1;
/* Computing 2nd power */
		d__2 = bkk;
		d__1 = akk * (d__2 * d__2);
		a[i__2].r = d__1, a[i__2].i = 0.;
/* L40: */
	    }
	}
    }
    return;

/*     End of ZHEGS2 */

} /* whegs2_ */

