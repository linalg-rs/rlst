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

/* > \brief \b ZLAQHP scales a Hermitian matrix stored in packed form. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZLAQHP + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zlaqhp.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zlaqhp.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zlaqhp.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZLAQHP( UPLO, N, AP, S, SCOND, AMAX, EQUED ) */

/*       CHARACTER          EQUED, UPLO */
/*       INTEGER            N */
/*       DOUBLE PRECISION   AMAX, SCOND */
/*       DOUBLE PRECISION   S( * ) */
/*       COMPLEX*16         AP( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZLAQHP equilibrates a Hermitian matrix A using the scaling factors */
/* > in the vector S. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          Specifies whether the upper or lower triangular part of the */
/* >          Hermitian matrix A is stored. */
/* >          = 'U':  Upper triangular */
/* >          = 'L':  Lower triangular */
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
/* >          On entry, the upper or lower triangle of the Hermitian matrix */
/* >          A, packed columnwise in a linear array.  The j-th column of A */
/* >          is stored in the array AP as follows: */
/* >          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j; */
/* >          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n. */
/* > */
/* >          On exit, the equilibrated matrix:  diag(S) * A * diag(S), in */
/* >          the same storage format as A. */
/* > \endverbatim */
/* > */
/* > \param[in] S */
/* > \verbatim */
/* >          S is DOUBLE PRECISION array, dimension (N) */
/* >          The scale factors for A. */
/* > \endverbatim */
/* > */
/* > \param[in] SCOND */
/* > \verbatim */
/* >          SCOND is DOUBLE PRECISION */
/* >          Ratio of the smallest S(i) to the largest S(i). */
/* > \endverbatim */
/* > */
/* > \param[in] AMAX */
/* > \verbatim */
/* >          AMAX is DOUBLE PRECISION */
/* >          Absolute value of largest matrix entry. */
/* > \endverbatim */
/* > */
/* > \param[out] EQUED */
/* > \verbatim */
/* >          EQUED is CHARACTER*1 */
/* >          Specifies whether or not equilibration was done. */
/* >          = 'N':  No equilibration. */
/* >          = 'Y':  Equilibration was done, i.e., A has been replaced by */
/* >                  diag(S) * A * diag(S). */
/* > \endverbatim */

/* > \par Internal Parameters: */
/*  ========================= */
/* > */
/* > \verbatim */
/* >  THRESH is a threshold value used to decide if scaling should be done */
/* >  based on the ratio of the scaling factors.  If SCOND < THRESH, */
/* >  scaling is done. */
/* > */
/* >  LARGE and SMALL are threshold values used to decide if scaling should */
/* >  be done based on the absolute size of the largest matrix element. */
/* >  If AMAX > LARGE or AMAX < SMALL, scaling is done. */
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
void  wlaqhp_(char *uplo, integer *n, quadcomplex *ap, 
	quadreal *s, quadreal *scond, quadreal *amax, char *equed)
{
    /* System generated locals */
    integer i__1, i__2, i__3, i__4;
    quadreal d__1;
    quadcomplex z__1;

    /* Local variables */
    integer i__, j, jc;
    quadreal cj, large;
    extern logical lsame_(char *, char *);
    quadreal small;
    extern quadreal qlamch_(char *);


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Quick return if possible */

    /* Parameter adjustments */
    --s;
    --ap;

    /* Function Body */
    if (*n <= 0) {
	*(unsigned char *)equed = 'N';
	return;
    }

/*     Initialize LARGE and SMALL. */

    small = qlamch_("Safe minimum") / qlamch_("Precision");
    large = 1. / small;

    if (*scond >= .1 && *amax >= small && *amax <= large) {

/*        No equilibration */

	*(unsigned char *)equed = 'N';
    } else {

/*        Replace A by diag(S) * A * diag(S). */

	if (lsame_(uplo, "U")) {

/*           Upper triangle of A is stored. */

	    jc = 1;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		cj = s[j];
		i__2 = j - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    i__3 = jc + i__ - 1;
		    d__1 = cj * s[i__];
		    i__4 = jc + i__ - 1;
		    z__1.r = d__1 * ap[i__4].r, z__1.i = d__1 * ap[i__4].i;
		    ap[i__3].r = z__1.r, ap[i__3].i = z__1.i;
/* L10: */
		}
		i__2 = jc + j - 1;
		i__3 = jc + j - 1;
		d__1 = cj * cj * ap[i__3].r;
		ap[i__2].r = d__1, ap[i__2].i = 0.;
		jc += j;
/* L20: */
	    }
	} else {

/*           Lower triangle of A is stored. */

	    jc = 1;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		cj = s[j];
		i__2 = jc;
		i__3 = jc;
		d__1 = cj * cj * ap[i__3].r;
		ap[i__2].r = d__1, ap[i__2].i = 0.;
		i__2 = *n;
		for (i__ = j + 1; i__ <= i__2; ++i__) {
		    i__3 = jc + i__ - j;
		    d__1 = cj * s[i__];
		    i__4 = jc + i__ - j;
		    z__1.r = d__1 * ap[i__4].r, z__1.i = d__1 * ap[i__4].i;
		    ap[i__3].r = z__1.r, ap[i__3].i = z__1.i;
/* L30: */
		}
		jc = jc + *n - j + 1;
/* L40: */
	    }
	}
	*(unsigned char *)equed = 'Y';
    }

    return;

/*     End of ZLAQHP */

} /* wlaqhp_ */

