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

static halfreal c_b6 = 1.;
static halfreal c_b7 = 0.;

/* > \brief \b ZLACRM multiplies a complex matrix by a square doublereal matrix. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZLACRM + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zlacrm.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zlacrm.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zlacrm.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZLACRM( M, N, A, LDA, B, LDB, C, LDC, RWORK ) */

/*       INTEGER            LDA, LDB, LDC, M, N */
/*       DOUBLE PRECISION   B( LDB, * ), RWORK( * ) */
/*       COMPLEX*16         A( LDA, * ), C( LDC, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZLACRM performs a very simple matrix-matrix multiplication: */
/* >          C := A * B, */
/* > where A is M by N and complex; B is N by N and doublereal; */
/* > C is M by N and complex. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of rows of the matrix A and of the matrix C. */
/* >          M >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of columns and rows of the matrix B and */
/* >          the number of columns of the matrix C. */
/* >          N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] A */
/* > \verbatim */
/* >          A is COMPLEX*16 array, dimension (LDA, N) */
/* >          On entry, A contains the M by N matrix A. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A. LDA >=f2cmax(1,M). */
/* > \endverbatim */
/* > */
/* > \param[in] B */
/* > \verbatim */
/* >          B is DOUBLE PRECISION array, dimension (LDB, N) */
/* >          On entry, B contains the N by N matrix B. */
/* > \endverbatim */
/* > */
/* > \param[in] LDB */
/* > \verbatim */
/* >          LDB is INTEGER */
/* >          The leading dimension of the array B. LDB >=f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[out] C */
/* > \verbatim */
/* >          C is COMPLEX*16 array, dimension (LDC, N) */
/* >          On exit, C contains the M by N matrix C. */
/* > \endverbatim */
/* > */
/* > \param[in] LDC */
/* > \verbatim */
/* >          LDC is INTEGER */
/* >          The leading dimension of the array C. LDC >=f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[out] RWORK */
/* > \verbatim */
/* >          RWORK is DOUBLE PRECISION array, dimension (2*M*N) */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complex16OTHERauxiliary */

/*  ===================================================================== */
void  klacrm_(integer *m, integer *n, halfcomplex *a, 
	integer *lda, halfreal *b, integer *ldb, halfcomplex *c__, 
	integer *ldc, halfreal *rwork)
{
    /* System generated locals */
    integer b_dim1, b_offset, a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, 
	    i__3, i__4, i__5;
    halfreal d__1;
    halfcomplex z__1;

    /* Local variables */
    integer i__, j, l;
    extern void  hgemm_(char *, char *, integer *, integer *, 
	    integer *, halfreal *, halfreal *, integer *, halfreal *, 
	    integer *, halfreal *, halfreal *, integer *);


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Quick return if possible. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --rwork;

    /* Function Body */
    if (*m == 0 || *n == 0) {
	return;
    }

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * a_dim1;
	    rwork[(j - 1) * *m + i__] = a[i__3].r;
/* L10: */
	}
/* L20: */
    }

    l = *m * *n + 1;
    hgemm_("N", "N", m, n, n, &c_b6, &rwork[1], m, &b[b_offset], ldb, &c_b7, &
	    rwork[l], m);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * c_dim1;
	    i__4 = l + (j - 1) * *m + i__ - 1;
	    c__[i__3].r = rwork[i__4], c__[i__3].i = 0.;
/* L30: */
	}
/* L40: */
    }

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    rwork[(j - 1) * *m + i__] = d_imag(&a[i__ + j * a_dim1]);
/* L50: */
	}
/* L60: */
    }
    hgemm_("N", "N", m, n, n, &c_b6, &rwork[1], m, &b[b_offset], ldb, &c_b7, &
	    rwork[l], m);
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * c_dim1;
	    i__4 = i__ + j * c_dim1;
	    d__1 = c__[i__4].r;
	    i__5 = l + (j - 1) * *m + i__ - 1;
	    z__1.r = d__1, z__1.i = rwork[i__5];
	    c__[i__3].r = z__1.r, c__[i__3].i = z__1.i;
/* L70: */
	}
/* L80: */
    }

    return;

/*     End of ZLACRM */

} /* klacrm_ */

