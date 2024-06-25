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
static integer c__1 = 1;

/* > \brief \b ZHPGST */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZHPGST + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zhpgst.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zhpgst.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zhpgst.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZHPGST( ITYPE, UPLO, N, AP, BP, INFO ) */

/*       CHARACTER          UPLO */
/*       INTEGER            INFO, ITYPE, N */
/*       COMPLEX*16         AP( * ), BP( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZHPGST reduces a complex Hermitian-definite generalized */
/* > eigenproblem to standard form, using packed storage. */
/* > */
/* > If ITYPE = 1, the problem is A*x = lambda*B*x, */
/* > and A is overwritten by inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H) */
/* > */
/* > If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or */
/* > B*A*x = lambda*x, and A is overwritten by U*A*U**H or L**H*A*L. */
/* > */
/* > B must have been previously factorized as U**H*U or L*L**H by ZPPTRF. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] ITYPE */
/* > \verbatim */
/* >          ITYPE is INTEGER */
/* >          = 1: compute inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H); */
/* >          = 2 or 3: compute U*A*U**H or L**H*A*L. */
/* > \endverbatim */
/* > */
/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          = 'U':  Upper triangle of A is stored and B is factored as */
/* >                  U**H*U; */
/* >          = 'L':  Lower triangle of A is stored and B is factored as */
/* >                  L*L**H. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrices A and B.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] AP */
/* > \verbatim */
/* >          AP is COMPLEX*16 array, dimension (N*(N+1)/2) */
/* >          On entry, the upper or lower triangle of the Hermitian matrix */
/* >          A, packed columnwise in a linear array.  The j-th column of A */
/* >          is stored in the array AP as follows: */
/* >          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j; */
/* >          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n. */
/* > */
/* >          On exit, if INFO = 0, the transformed matrix, stored in the */
/* >          same format as A. */
/* > \endverbatim */
/* > */
/* > \param[in] BP */
/* > \verbatim */
/* >          BP is COMPLEX*16 array, dimension (N*(N+1)/2) */
/* >          The triangular factor from the Cholesky factorization of B, */
/* >          stored in the same format as A, as returned by ZPPTRF. */
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

/* > \date December 2016 */

/* > \ingroup complex16OTHERcomputational */

/*  ===================================================================== */
void  khpgst_(integer *itype, char *uplo, integer *n, 
	halfcomplex *ap, halfcomplex *bp, integer *info)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    halfreal d__1, d__2;
    halfcomplex z__1, z__2, z__3;

    /* Local variables */
    integer j, k, j1, k1, jj, kk;
    halfcomplex ct;
    halfreal ajj;
    integer j1j1;
    halfreal akk;
    integer k1k1;
    halfreal bjj, bkk;
    extern void  khpr2_(char *, integer *, halfcomplex *, 
	    halfcomplex *, integer *, halfcomplex *, integer *, 
	    halfcomplex *);
    extern logical lsame_(char *, char *);
    extern /* Double Complex */ VOID whotc_(halfcomplex *, integer *, 
	    halfcomplex *, integer *, halfcomplex *, integer *);
    logical upper;
    extern void  khpmv_(char *, integer *, halfcomplex *, 
	    halfcomplex *, halfcomplex *, integer *, halfcomplex *, 
	    halfcomplex *, integer *), kaxpy_(integer *, 
	    halfcomplex *, halfcomplex *, integer *, halfcomplex *, 
	    integer *), ktpmv_(char *, char *, char *, integer *, 
	    halfcomplex *, halfcomplex *, integer *), ktpsv_(char *, char *, char *, integer *, halfcomplex *
	    , halfcomplex *, integer *), xerbla_(
	    char *, integer *), whscal_(integer *, halfreal *, 
	    halfcomplex *, integer *);


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    --bp;
    --ap;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (*itype < 1 || *itype > 3) {
	*info = -1;
    } else if (! upper && ! lsame_(uplo, "L")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZHPGST", &i__1);
	return;
    }

    if (*itype == 1) {
	if (upper) {

/*           Compute inv(U**H)*A*inv(U) */

/*           J1 and JJ are the indices of A(1,j) and A(j,j) */

	    jj = 0;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		j1 = jj + 1;
		jj += j;

/*              Compute the j-th column of the upper triangle of A */

		i__2 = jj;
		i__3 = jj;
		d__1 = ap[i__3].r;
		ap[i__2].r = d__1, ap[i__2].i = 0.;
		i__2 = jj;
		bjj = bp[i__2].r;
		ktpsv_(uplo, "Conjugate transpose", "Non-unit", &j, &bp[1], &
			ap[j1], &c__1);
		i__2 = j - 1;
		z__1.r = -1., z__1.i = -0.;
		khpmv_(uplo, &i__2, &z__1, &ap[1], &bp[j1], &c__1, &c_b1, &ap[
			j1], &c__1);
		i__2 = j - 1;
		d__1 = 1. / bjj;
		whscal_(&i__2, &d__1, &ap[j1], &c__1);
		i__2 = jj;
		i__3 = jj;
		i__4 = j - 1;
		whotc_(&z__3, &i__4, &ap[j1], &c__1, &bp[j1], &c__1);
		z__2.r = ap[i__3].r - z__3.r, z__2.i = ap[i__3].i - z__3.i;
		z__1.r = z__2.r / bjj, z__1.i = z__2.i / bjj;
		ap[i__2].r = z__1.r, ap[i__2].i = z__1.i;
/* L10: */
	    }
	} else {

/*           Compute inv(L)*A*inv(L**H) */

/*           KK and K1K1 are the indices of A(k,k) and A(k+1,k+1) */

	    kk = 1;
	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {
		k1k1 = kk + *n - k + 1;

/*              Update the lower triangle of A(k:n,k:n) */

		i__2 = kk;
		akk = ap[i__2].r;
		i__2 = kk;
		bkk = bp[i__2].r;
/* Computing 2nd power */
		d__1 = bkk;
		akk /= d__1 * d__1;
		i__2 = kk;
		ap[i__2].r = akk, ap[i__2].i = 0.;
		if (k < *n) {
		    i__2 = *n - k;
		    d__1 = 1. / bkk;
		    whscal_(&i__2, &d__1, &ap[kk + 1], &c__1);
		    d__1 = akk * -.5;
		    ct.r = d__1, ct.i = 0.;
		    i__2 = *n - k;
		    kaxpy_(&i__2, &ct, &bp[kk + 1], &c__1, &ap[kk + 1], &c__1)
			    ;
		    i__2 = *n - k;
		    z__1.r = -1., z__1.i = -0.;
		    khpr2_(uplo, &i__2, &z__1, &ap[kk + 1], &c__1, &bp[kk + 1]
			    , &c__1, &ap[k1k1]);
		    i__2 = *n - k;
		    kaxpy_(&i__2, &ct, &bp[kk + 1], &c__1, &ap[kk + 1], &c__1)
			    ;
		    i__2 = *n - k;
		    ktpsv_(uplo, "No transpose", "Non-unit", &i__2, &bp[k1k1],
			     &ap[kk + 1], &c__1);
		}
		kk = k1k1;
/* L20: */
	    }
	}
    } else {
	if (upper) {

/*           Compute U*A*U**H */

/*           K1 and KK are the indices of A(1,k) and A(k,k) */

	    kk = 0;
	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {
		k1 = kk + 1;
		kk += k;

/*              Update the upper triangle of A(1:k,1:k) */

		i__2 = kk;
		akk = ap[i__2].r;
		i__2 = kk;
		bkk = bp[i__2].r;
		i__2 = k - 1;
		ktpmv_(uplo, "No transpose", "Non-unit", &i__2, &bp[1], &ap[
			k1], &c__1);
		d__1 = akk * .5;
		ct.r = d__1, ct.i = 0.;
		i__2 = k - 1;
		kaxpy_(&i__2, &ct, &bp[k1], &c__1, &ap[k1], &c__1);
		i__2 = k - 1;
		khpr2_(uplo, &i__2, &c_b1, &ap[k1], &c__1, &bp[k1], &c__1, &
			ap[1]);
		i__2 = k - 1;
		kaxpy_(&i__2, &ct, &bp[k1], &c__1, &ap[k1], &c__1);
		i__2 = k - 1;
		whscal_(&i__2, &bkk, &ap[k1], &c__1);
		i__2 = kk;
/* Computing 2nd power */
		d__2 = bkk;
		d__1 = akk * (d__2 * d__2);
		ap[i__2].r = d__1, ap[i__2].i = 0.;
/* L30: */
	    }
	} else {

/*           Compute L**H *A*L */

/*           JJ and J1J1 are the indices of A(j,j) and A(j+1,j+1) */

	    jj = 1;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		j1j1 = jj + *n - j + 1;

/*              Compute the j-th column of the lower triangle of A */

		i__2 = jj;
		ajj = ap[i__2].r;
		i__2 = jj;
		bjj = bp[i__2].r;
		i__2 = jj;
		d__1 = ajj * bjj;
		i__3 = *n - j;
		whotc_(&z__2, &i__3, &ap[jj + 1], &c__1, &bp[jj + 1], &c__1);
		z__1.r = d__1 + z__2.r, z__1.i = z__2.i;
		ap[i__2].r = z__1.r, ap[i__2].i = z__1.i;
		i__2 = *n - j;
		whscal_(&i__2, &bjj, &ap[jj + 1], &c__1);
		i__2 = *n - j;
		khpmv_(uplo, &i__2, &c_b1, &ap[j1j1], &bp[jj + 1], &c__1, &
			c_b1, &ap[jj + 1], &c__1);
		i__2 = *n - j + 1;
		ktpmv_(uplo, "Conjugate transpose", "Non-unit", &i__2, &bp[jj]
			, &ap[jj], &c__1);
		jj = j1j1;
/* L40: */
	    }
	}
    }
    return;

/*     End of ZHPGST */

} /* khpgst_ */

