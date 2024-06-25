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

static quadreal c_b8 = 1.;
static integer c__1 = 1;

/* > \brief \b ZPPTRI */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZPPTRI + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zpptri.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zpptri.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zpptri.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZPPTRI( UPLO, N, AP, INFO ) */

/*       CHARACTER          UPLO */
/*       INTEGER            INFO, N */
/*       COMPLEX*16         AP( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZPPTRI computes the inverse of a complex Hermitian positive definite */
/* > matrix A using the Cholesky factorization A = U**H*U or A = L*L**H */
/* > computed by ZPPTRF. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          = 'U':  Upper triangular factor is stored in AP; */
/* >          = 'L':  Lower triangular factor is stored in AP. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] AP */
/* > \verbatim */
/* >          AP is COMPLEX*16 array, dimension (N*(N+1)/2) */
/* >          On entry, the triangular factor U or L from the Cholesky */
/* >          factorization A = U**H*U or A = L*L**H, packed columnwise as */
/* >          a linear array.  The j-th column of U or L is stored in the */
/* >          array AP as follows: */
/* >          if UPLO = 'U', AP(i + (j-1)*j/2) = U(i,j) for 1<=i<=j; */
/* >          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = L(i,j) for j<=i<=n. */
/* > */
/* >          On exit, the upper or lower triangle of the (Hermitian) */
/* >          inverse of A, overwriting the input factor U or L. */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value */
/* >          > 0:  if INFO = i, the (i,i) element of the factor U or L is */
/* >                zero, and the inverse could not be computed. */
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
void  wpptri_(char *uplo, integer *n, quadcomplex *ap, 
	integer *info)
{
    /* System generated locals */
    integer i__1, i__2, i__3;
    quadreal d__1;
    quadcomplex z__1;

    /* Local variables */
    integer j, jc, jj;
    quadreal ajj;
    integer jjn;
    extern void  whpr_(char *, integer *, quadreal *, 
	    quadcomplex *, integer *, quadcomplex *);
    extern logical lsame_(char *, char *);
    extern /* Double Complex */ VOID wqotc_(quadcomplex *, integer *, 
	    quadcomplex *, integer *, quadcomplex *, integer *);
    logical upper;
    extern void  wtpmv_(char *, char *, char *, integer *, 
	    quadcomplex *, quadcomplex *, integer *), xerbla_(char *, integer *), wqscal_(integer *, 
	    quadreal *, quadcomplex *, integer *), wtptri_(char *, char *,
	     integer *, quadcomplex *, integer *);


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    --ap;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZPPTRI", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return;
    }

/*     Invert the triangular Cholesky factor U or L. */

    wtptri_(uplo, "Non-unit", n, &ap[1], info);
    if (*info > 0) {
	return;
    }
    if (upper) {

/*        Compute the product inv(U) * inv(U)**H. */

	jj = 0;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    jc = jj + 1;
	    jj += j;
	    if (j > 1) {
		i__2 = j - 1;
		whpr_("Upper", &i__2, &c_b8, &ap[jc], &c__1, &ap[1]);
	    }
	    i__2 = jj;
	    ajj = ap[i__2].r;
	    wqscal_(&j, &ajj, &ap[jc], &c__1);
/* L10: */
	}

    } else {

/*        Compute the product inv(L)**H * inv(L). */

	jj = 1;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    jjn = jj + *n - j + 1;
	    i__2 = jj;
	    i__3 = *n - j + 1;
	    wqotc_(&z__1, &i__3, &ap[jj], &c__1, &ap[jj], &c__1);
	    d__1 = z__1.r;
	    ap[i__2].r = d__1, ap[i__2].i = 0.;
	    if (j < *n) {
		i__2 = *n - j;
		wtpmv_("Lower", "Conjugate transpose", "Non-unit", &i__2, &ap[
			jjn], &ap[jj + 1], &c__1);
	    }
	    jj = jjn;
/* L20: */
	}
    }

    return;

/*     End of ZPPTRI */

} /* wpptri_ */

