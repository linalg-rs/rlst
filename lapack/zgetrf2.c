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

/* > \brief \b ZGETRF2 */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/*  Definition: */
/*  =========== */

/*        SUBROUTINE ZGETRF2( M, N, A, LDA, IPIV, INFO ) */

/*       INTEGER            INFO, LDA, M, N */
/*       INTEGER            IPIV( * ) */
/*       COMPLEX*16         A( LDA, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZGETRF2 computes an LU factorization of a general M-by-N matrix A */
/* > using partial pivoting with row interchanges. */
/* > */
/* > The factorization has the form */
/* >    A = P * L * U */
/* > where P is a permutation matrix, L is lower triangular with unit */
/* > diagonal elements (lower trapezoidal if m > n), and U is upper */
/* > triangular (upper trapezoidal if m < n). */
/* > */
/* > This is the recursive version of the algorithm. It divides */
/* > the matrix into four submatrices: */
/* > */
/* >        [  A11 | A12  ]  where A11 is n1 by n1 and A22 is n2 by n2 */
/* >    A = [ -----|----- ]  with n1 = f2cmin(m,n)/2 */
/* >        [  A21 | A22  ]       n2 = n-n1 */
/* > */
/* >                                       [ A11 ] */
/* > The subroutine calls itself to factor [ --- ], */
/* >                                       [ A12 ] */
/* >                 [ A12 ] */
/* > do the swaps on [ --- ], solve A12, update A22, */
/* >                 [ A22 ] */
/* > */
/* > then calls itself to factor A22 and do the swaps on A21. */
/* > */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of rows of the matrix A.  M >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of columns of the matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is COMPLEX*16 array, dimension (LDA,N) */
/* >          On entry, the M-by-N matrix to be factored. */
/* >          On exit, the factors L and U from the factorization */
/* >          A = P*L*U; the unit diagonal elements of L are not stored. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,M). */
/* > \endverbatim */
/* > */
/* > \param[out] IPIV */
/* > \verbatim */
/* >          IPIV is INTEGER array, dimension (f2cmin(M,N)) */
/* >          The pivot indices; for 1 <= i <= f2cmin(M,N), row i of the */
/* >          matrix was interchanged with row IPIV(i). */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value */
/* >          > 0:  if INFO = i, U(i,i) is exactly zero. The factorization */
/* >                has been completed, but the factor U is exactly */
/* >                singular, and division by zero will occur if it is used */
/* >                to solve a system of equations. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date June 2016 */

/* > \ingroup complex16GEcomputational */

/*  ===================================================================== */
void  zgetrf2_(integer *m, integer *n, doublecomplex *a, 
	integer *lda, integer *ipiv, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    doublecomplex z__1;

    /* Local variables */
    integer i__, n1, n2;
    doublecomplex temp;
    integer iinfo;
    doublereal sfmin;
    extern void  zscal_(integer *, doublecomplex *, 
	    doublecomplex *, integer *), zgemm_(char *, char *, integer *, 
	    integer *, integer *, doublecomplex *, doublecomplex *, integer *,
	     doublecomplex *, integer *, doublecomplex *, doublecomplex *, 
	    integer *), ztrsm_(char *, char *, char *, char *,
	     integer *, integer *, doublecomplex *, doublecomplex *, integer *
	    , doublecomplex *, integer *);
    extern doublereal dlamch_(char *);
    extern void  xerbla_(char *, integer *);
    extern integer izamax_(integer *, doublecomplex *, integer *);
    extern void  zlaswp_(integer *, doublecomplex *, integer *,
	     integer *, integer *, integer *, integer *);


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     June 2016 */


/*  ===================================================================== */


/*     Test the input parameters */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;

    /* Function Body */
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < f2cmax(1,*m)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZGETRF2", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0) {
	return;
    }
    if (*m == 1) {

/*        Use unblocked code for one row case */
/*        Just need to handle IPIV and INFO */

	ipiv[1] = 1;
	i__1 = a_dim1 + 1;
	if (a[i__1].r == 0. && a[i__1].i == 0.) {
	    *info = 1;
	}

    } else if (*n == 1) {

/*        Use unblocked code for one column case */


/*        Compute machine safe minimum */

	sfmin = dlamch_("S");

/*        Find pivot and test for singularity */

	i__ = izamax_(m, &a[a_dim1 + 1], &c__1);
	ipiv[1] = i__;
	i__1 = i__ + a_dim1;
	if (a[i__1].r != 0. || a[i__1].i != 0.) {

/*           Apply the interchange */

	    if (i__ != 1) {
		i__1 = a_dim1 + 1;
		temp.r = a[i__1].r, temp.i = a[i__1].i;
		i__1 = a_dim1 + 1;
		i__2 = i__ + a_dim1;
		a[i__1].r = a[i__2].r, a[i__1].i = a[i__2].i;
		i__1 = i__ + a_dim1;
		a[i__1].r = temp.r, a[i__1].i = temp.i;
	    }

/*           Compute elements 2:M of the column */

	    if (z_abs(&a[a_dim1 + 1]) >= sfmin) {
		i__1 = *m - 1;
		z_div(&z__1, &c_b1, &a[a_dim1 + 1]);
		zscal_(&i__1, &z__1, &a[a_dim1 + 2], &c__1);
	    } else {
		i__1 = *m - 1;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    i__2 = i__ + 1 + a_dim1;
		    z_div(&z__1, &a[i__ + 1 + a_dim1], &a[a_dim1 + 1]);
		    a[i__2].r = z__1.r, a[i__2].i = z__1.i;
/* L10: */
		}
	    }

	} else {
	    *info = 1;
	}
    } else {

/*        Use recursive code */

	n1 = f2cmin(*m,*n) / 2;
	n2 = *n - n1;

/*               [ A11 ] */
/*        Factor [ --- ] */
/*               [ A21 ] */

	zgetrf2_(m, &n1, &a[a_offset], lda, &ipiv[1], &iinfo);
	if (*info == 0 && iinfo > 0) {
	    *info = iinfo;
	}

/*                              [ A12 ] */
/*        Apply interchanges to [ --- ] */
/*                              [ A22 ] */

	zlaswp_(&n2, &a[(n1 + 1) * a_dim1 + 1], lda, &c__1, &n1, &ipiv[1], &
		c__1);

/*        Solve A12 */

	ztrsm_("L", "L", "N", "U", &n1, &n2, &c_b1, &a[a_offset], lda, &a[(n1 
		+ 1) * a_dim1 + 1], lda);

/*        Update A22 */

	i__1 = *m - n1;
	z__1.r = -1., z__1.i = -0.;
	zgemm_("N", "N", &i__1, &n2, &n1, &z__1, &a[n1 + 1 + a_dim1], lda, &a[
		(n1 + 1) * a_dim1 + 1], lda, &c_b1, &a[n1 + 1 + (n1 + 1) * 
		a_dim1], lda);

/*        Factor A22 */

	i__1 = *m - n1;
	zgetrf2_(&i__1, &n2, &a[n1 + 1 + (n1 + 1) * a_dim1], lda, &ipiv[n1 + 
		1], &iinfo);

/*        Adjust INFO and the pivot indices */

	if (*info == 0 && iinfo > 0) {
	    *info = iinfo + n1;
	}
	i__1 = f2cmin(*m,*n);
	for (i__ = n1 + 1; i__ <= i__1; ++i__) {
	    ipiv[i__] += n1;
/* L20: */
	}

/*        Apply interchanges to A21 */

	i__1 = n1 + 1;
	i__2 = f2cmin(*m,*n);
	zlaswp_(&n1, &a[a_dim1 + 1], lda, &i__1, &i__2, &ipiv[1], &c__1);

    }
    return;

/*     End of ZGETRF2 */

} /* zgetrf2_ */
