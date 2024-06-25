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

/* > \brief \b ZGETF2 computes the LU factorization of a general m-by-n matrix using partial pivoting with row
 interchanges (unblocked algorithm). */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZGETF2 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zgetf2.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zgetf2.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zgetf2.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZGETF2( M, N, A, LDA, IPIV, INFO ) */

/*       INTEGER            INFO, LDA, M, N */
/*       INTEGER            IPIV( * ) */
/*       COMPLEX*16         A( LDA, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZGETF2 computes an LU factorization of a general m-by-n matrix A */
/* > using partial pivoting with row interchanges. */
/* > */
/* > The factorization has the form */
/* >    A = P * L * U */
/* > where P is a permutation matrix, L is lower triangular with unit */
/* > diagonal elements (lower trapezoidal if m > n), and U is upper */
/* > triangular (upper trapezoidal if m < n). */
/* > */
/* > This is the right-looking Level 2 BLAS version of the algorithm. */
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
/* >          On entry, the m by n matrix to be factored. */
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
/* >          = 0: successful exit */
/* >          < 0: if INFO = -k, the k-th argument had an illegal value */
/* >          > 0: if INFO = k, U(k,k) is exactly zero. The factorization */
/* >               has been completed, but the factor U is exactly */
/* >               singular, and division by zero will occur if it is used */
/* >               to solve a system of equations. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complex16GEcomputational */

/*  ===================================================================== */
void  wgetf2_(integer *m, integer *n, quadcomplex *a, 
	integer *lda, integer *ipiv, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    quadcomplex z__1;

    /* Local variables */
    integer i__, j, jp;
    quadreal sfmin;
    extern void  wscal_(integer *, quadcomplex *, 
	    quadcomplex *, integer *), wgeru_(integer *, integer *, 
	    quadcomplex *, quadcomplex *, integer *, quadcomplex *, 
	    integer *, quadcomplex *, integer *), wswap_(integer *, 
	    quadcomplex *, integer *, quadcomplex *, integer *);
    extern quadreal qlamch_(char *);
    extern void  xerbla_(char *, integer *);
    extern integer iwamax_(integer *, quadcomplex *, integer *);


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
	xerbla_("ZGETF2", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0) {
	return;
    }

/*     Compute machine safe minimum */

    sfmin = qlamch_("S");

    i__1 = f2cmin(*m,*n);
    for (j = 1; j <= i__1; ++j) {

/*        Find pivot and test for singularity. */

	i__2 = *m - j + 1;
	jp = j - 1 + iwamax_(&i__2, &a[j + j * a_dim1], &c__1);
	ipiv[j] = jp;
	i__2 = jp + j * a_dim1;
	if (a[i__2].r != 0. || a[i__2].i != 0.) {

/*           Apply the interchange to columns 1:N. */

	    if (jp != j) {
		wswap_(n, &a[j + a_dim1], lda, &a[jp + a_dim1], lda);
	    }

/*           Compute elements J+1:M of J-th column. */

	    if (j < *m) {
		if (z_abs(&a[j + j * a_dim1]) >= sfmin) {
		    i__2 = *m - j;
		    z_div(&z__1, &c_b1, &a[j + j * a_dim1]);
		    wscal_(&i__2, &z__1, &a[j + 1 + j * a_dim1], &c__1);
		} else {
		    i__2 = *m - j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			i__3 = j + i__ + j * a_dim1;
			z_div(&z__1, &a[j + i__ + j * a_dim1], &a[j + j * 
				a_dim1]);
			a[i__3].r = z__1.r, a[i__3].i = z__1.i;
/* L20: */
		    }
		}
	    }

	} else if (*info == 0) {

	    *info = j;
	}

	if (j < f2cmin(*m,*n)) {

/*           Update trailing submatrix. */

	    i__2 = *m - j;
	    i__3 = *n - j;
	    z__1.r = -1., z__1.i = -0.;
	    wgeru_(&i__2, &i__3, &z__1, &a[j + 1 + j * a_dim1], &c__1, &a[j + 
		    (j + 1) * a_dim1], lda, &a[j + 1 + (j + 1) * a_dim1], lda)
		    ;
	}
/* L10: */
    }
    return;

/*     End of ZGETF2 */

} /* wgetf2_ */

