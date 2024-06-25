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

static integer c__1 = 1;
static integer c__2 = 2;
static integer c__0 = 0;

/* > \brief \b DLASQ1 computes the singular values of a doublereal square bidiagonal matrix. Used by sbdsqr. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DLASQ1 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasq1.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasq1.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasq1.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DLASQ1( N, D, E, WORK, INFO ) */

/*       INTEGER            INFO, N */
/*       DOUBLE PRECISION   D( * ), E( * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DLASQ1 computes the singular values of a doublereal N-by-N bidiagonal */
/* > matrix with diagonal D and off-diagonal E. The singular values */
/* > are computed to high relative accuracy, in the absence of */
/* > denormalization, underflow and overflow. The algorithm was first */
/* > presented in */
/* > */
/* > "Accurate singular values and differential qd algorithms" by K. V. */
/* > Fernando and B. N. Parlett, Numer. Math., Vol-67, No. 2, pp. 191-230, */
/* > 1994, */
/* > */
/* > and the present implementation is described in "An implementation of */
/* > the dqds Algorithm (Positive Case)", LAPACK Working Note. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >        The number of rows and columns in the matrix. N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] D */
/* > \verbatim */
/* >          D is DOUBLE PRECISION array, dimension (N) */
/* >        On entry, D contains the diagonal elements of the */
/* >        bidiagonal matrix whose SVD is desired. On normal exit, */
/* >        D contains the singular values in decreasing order. */
/* > \endverbatim */
/* > */
/* > \param[in,out] E */
/* > \verbatim */
/* >          E is DOUBLE PRECISION array, dimension (N) */
/* >        On entry, elements E(1:N-1) contain the off-diagonal elements */
/* >        of the bidiagonal matrix whose SVD is desired. */
/* >        On exit, E is overwritten. */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is DOUBLE PRECISION array, dimension (4*N) */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >        = 0: successful exit */
/* >        < 0: if INFO = -i, the i-th argument had an illegal value */
/* >        > 0: the algorithm failed */
/* >             = 1, a split was marked by a positive value in E */
/* >             = 2, current block of Z not diagonalized after 100*N */
/* >                  iterations (in inner while loop)  On exit D and E */
/* >                  represent a matrix with the same singular values */
/* >                  which the calling subroutine could use to finish the */
/* >                  computation, or even feed back into DLASQ1 */
/* >             = 3, termination criterion of outer while loop not met */
/* >                  (program created more than N unreduced blocks) */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup auxOTHERcomputational */

/*  ===================================================================== */
void  qlasq1_(integer *n, quadreal *d__, quadreal *e, 
	quadreal *work, integer *info)
{
    /* System generated locals */
    integer i__1, i__2;
    quadreal d__1, d__2, d__3;

    /* Local variables */
    integer i__;
    quadreal eps;
    extern void  qlas2_(quadreal *, quadreal *, quadreal 
	    *, quadreal *, quadreal *);
    quadreal scale;
    integer iinfo;
    quadreal sigmn;
    extern void  qcopy_(integer *, quadreal *, integer *, 
	    quadreal *, integer *);
    quadreal sigmx;
    extern void  qlasq2_(integer *, quadreal *, integer *);
    extern quadreal qlamch_(char *);
    extern void  qlascl_(char *, integer *, integer *, 
	    quadreal *, quadreal *, integer *, integer *, quadreal *, 
	    integer *, integer *);
    quadreal safmin;
    extern void  xerbla_(char *, integer *), qlasrt_(
	    char *, integer *, quadreal *, integer *);


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    --work;
    --e;
    --d__;

    /* Function Body */
    *info = 0;
    if (*n < 0) {
	*info = -1;
	i__1 = -(*info);
	xerbla_("DLASQ1", &i__1);
	return;
    } else if (*n == 0) {
	return;
    } else if (*n == 1) {
	d__[1] = abs(d__[1]);
	return;
    } else if (*n == 2) {
	qlas2_(&d__[1], &e[1], &d__[2], &sigmn, &sigmx);
	d__[1] = sigmx;
	d__[2] = sigmn;
	return;
    }

/*     Estimate the largest singular value. */

    sigmx = 0.;
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	d__[i__] = (d__1 = d__[i__], abs(d__1));
/* Computing MAX */
	d__2 = sigmx, d__3 = (d__1 = e[i__], abs(d__1));
	sigmx = f2cmax(d__2,d__3);
/* L10: */
    }
    d__[*n] = (d__1 = d__[*n], abs(d__1));

/*     Early return if SIGMX is zero (matrix is already diagonal). */

    if (sigmx == 0.) {
	qlasrt_("D", n, &d__[1], &iinfo);
	return;
    }

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
	d__1 = sigmx, d__2 = d__[i__];
	sigmx = f2cmax(d__1,d__2);
/* L20: */
    }

/*     Copy D and E into WORK (in the Z format) and scale (squaring the */
/*     input data makes scaling by a power of the radix pointless). */

    eps = qlamch_("Precision");
    safmin = qlamch_("Safe minimum");
    scale = M(sqrt)(eps / safmin);
    qcopy_(n, &d__[1], &c__1, &work[1], &c__2);
    i__1 = *n - 1;
    qcopy_(&i__1, &e[1], &c__1, &work[2], &c__2);
    i__1 = (*n << 1) - 1;
    i__2 = (*n << 1) - 1;
    qlascl_("G", &c__0, &c__0, &sigmx, &scale, &i__1, &c__1, &work[1], &i__2, 
	    &iinfo);

/*     Compute the q's and e's. */

    i__1 = (*n << 1) - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing 2nd power */
	d__1 = work[i__];
	work[i__] = d__1 * d__1;
/* L30: */
    }
    work[*n * 2] = 0.;

    qlasq2_(n, &work[1], info);

    if (*info == 0) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d__[i__] = M(sqrt)(work[i__]);
/* L40: */
	}
	qlascl_("G", &c__0, &c__0, &scale, &sigmx, n, &c__1, &d__[1], n, &
		iinfo);
    } else if (*info == 2) {

/*     Maximum number of iterations exceeded.  Move data from WORK */
/*     into D and E so the calling subroutine can try to finish */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d__[i__] = M(sqrt)(work[(i__ << 1) - 1]);
	    e[i__] = M(sqrt)(work[i__ * 2]);
	}
	qlascl_("G", &c__0, &c__0, &scale, &sigmx, n, &c__1, &d__[1], n, &
		iinfo);
	qlascl_("G", &c__0, &c__0, &scale, &sigmx, n, &c__1, &e[1], n, &iinfo);
    }

    return;

/*     End of DLASQ1 */

} /* qlasq1_ */

