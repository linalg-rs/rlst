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

#define __LAPACK_PRECISION_SINGLE
#include "f2c.h"

/* Table of constant values */

static integer c__1 = 1;
static integer c_n1 = -1;
static real c_b16 = 1.f;
static real c_b19 = -1.f;

/* > \brief \b SGETRF */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download SGETRF + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/sgetrf.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/sgetrf.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/sgetrf.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE SGETRF( M, N, A, LDA, IPIV, INFO ) */

/*       INTEGER            INFO, LDA, M, N */
/*       INTEGER            IPIV( * ) */
/*       REAL               A( LDA, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > SGETRF computes an LU factorization of a general M-by-N matrix A */
/* > using partial pivoting with row interchanges. */
/* > */
/* > The factorization has the form */
/* >    A = P * L * U */
/* > where P is a permutation matrix, L is lower triangular with unit */
/* > diagonal elements (lower trapezoidal if m > n), and U is upper */
/* > triangular (upper trapezoidal if m < n). */
/* > */
/* > This is the right-looking Level 3 BLAS version of the algorithm. */
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
/* >          A is REAL array, dimension (LDA,N) */
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

/* > \date December 2016 */

/* > \ingroup realGEcomputational */

/*  ===================================================================== */
void  sgetrf_(integer *m, integer *n, real *a, integer *lda, 
	integer *ipiv, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;

    /* Local variables */
    integer i__, j, jb, nb, iinfo;
    extern void  sgemm_(char *, char *, integer *, integer *, 
	    integer *, real *, real *, integer *, real *, integer *, real *, 
	    real *, integer *), strsm_(char *, char *, char *,
	     char *, integer *, integer *, real *, real *, integer *, real *, 
	    integer *), xerbla_(char *, 
	    integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *, ftnlen, ftnlen);
    extern void  slaswp_(integer *, real *, integer *, integer 
	    *, integer *, integer *, integer *), sgetrf2_(integer *, integer *
	    , real *, integer *, integer *, integer *);


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
	xerbla_("SGETRF", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0) {
	return;
    }

/*     Determine the block size for this environment. */

    nb = ilaenv_(&c__1, "SGETRF", " ", m, n, &c_n1, &c_n1, (ftnlen)6, (ftnlen)
	    1);
    if (nb <= 1 || nb >= f2cmin(*m,*n)) {

/*        Use unblocked code. */

	sgetrf2_(m, n, &a[a_offset], lda, &ipiv[1], info);
    } else {

/*        Use blocked code. */

	i__1 = f2cmin(*m,*n);
	i__2 = nb;
	for (j = 1; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {
/* Computing MIN */
	    i__3 = f2cmin(*m,*n) - j + 1;
	    jb = f2cmin(i__3,nb);

/*           Factor diagonal and subdiagonal blocks and test for exact */
/*           singularity. */

	    i__3 = *m - j + 1;
	    sgetrf2_(&i__3, &jb, &a[j + j * a_dim1], lda, &ipiv[j], &iinfo);

/*           Adjust INFO and the pivot indices. */

	    if (*info == 0 && iinfo > 0) {
		*info = iinfo + j - 1;
	    }
/* Computing MIN */
	    i__4 = *m, i__5 = j + jb - 1;
	    i__3 = f2cmin(i__4,i__5);
	    for (i__ = j; i__ <= i__3; ++i__) {
		ipiv[i__] = j - 1 + ipiv[i__];
/* L10: */
	    }

/*           Apply interchanges to columns 1:J-1. */

	    i__3 = j - 1;
	    i__4 = j + jb - 1;
	    slaswp_(&i__3, &a[a_offset], lda, &j, &i__4, &ipiv[1], &c__1);

	    if (j + jb <= *n) {

/*              Apply interchanges to columns J+JB:N. */

		i__3 = *n - j - jb + 1;
		i__4 = j + jb - 1;
		slaswp_(&i__3, &a[(j + jb) * a_dim1 + 1], lda, &j, &i__4, &
			ipiv[1], &c__1);

/*              Compute block row of U. */

		i__3 = *n - j - jb + 1;
		strsm_("Left", "Lower", "No transpose", "Unit", &jb, &i__3, &
			c_b16, &a[j + j * a_dim1], lda, &a[j + (j + jb) * 
			a_dim1], lda);
		if (j + jb <= *m) {

/*                 Update trailing submatrix. */

		    i__3 = *m - j - jb + 1;
		    i__4 = *n - j - jb + 1;
		    sgemm_("No transpose", "No transpose", &i__3, &i__4, &jb, 
			    &c_b19, &a[j + jb + j * a_dim1], lda, &a[j + (j + 
			    jb) * a_dim1], lda, &c_b16, &a[j + jb + (j + jb) *
			     a_dim1], lda);
		}
	    }
/* L20: */
	}
    }
    return;

/*     End of SGETRF */

} /* sgetrf_ */

