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

static integer c__1 = 1;
static integer c__0 = 0;
static halfreal c_b8 = 1.;

/* > \brief \b DLASD8 finds the square roots of the roots of the secular equation, and stores, for each elemen
t in D, the distance to its two nearest poles. Used by sbdsdc. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DLASD8 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasd8.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasd8.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasd8.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DLASD8( ICOMPQ, K, D, Z, VF, VL, DIFL, DIFR, LDDIFR, */
/*                          DSIGMA, WORK, INFO ) */

/*       INTEGER            ICOMPQ, INFO, K, LDDIFR */
/*       DOUBLE PRECISION   D( * ), DIFL( * ), DIFR( LDDIFR, * ), */
/*      $                   DSIGMA( * ), VF( * ), VL( * ), WORK( * ), */
/*      $                   Z( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DLASD8 finds the square roots of the roots of the secular equation, */
/* > as defined by the values in DSIGMA and Z. It makes the appropriate */
/* > calls to DLASD4, and stores, for each  element in D, the distance */
/* > to its two nearest poles (elements in DSIGMA). It also updates */
/* > the arrays VF and VL, the first and last components of all the */
/* > right singular vectors of the original bidiagonal matrix. */
/* > */
/* > DLASD8 is called from DLASD6. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] ICOMPQ */
/* > \verbatim */
/* >          ICOMPQ is INTEGER */
/* >          Specifies whether singular vectors are to be computed in */
/* >          factored form in the calling routine: */
/* >          = 0: Compute singular values only. */
/* >          = 1: Compute singular vectors in factored form as well. */
/* > \endverbatim */
/* > */
/* > \param[in] K */
/* > \verbatim */
/* >          K is INTEGER */
/* >          The number of terms in the rational function to be solved */
/* >          by DLASD4.  K >= 1. */
/* > \endverbatim */
/* > */
/* > \param[out] D */
/* > \verbatim */
/* >          D is DOUBLE PRECISION array, dimension ( K ) */
/* >          On output, D contains the updated singular values. */
/* > \endverbatim */
/* > */
/* > \param[in,out] Z */
/* > \verbatim */
/* >          Z is DOUBLE PRECISION array, dimension ( K ) */
/* >          On entry, the first K elements of this array contain the */
/* >          components of the deflation-adjusted updating row vector. */
/* >          On exit, Z is updated. */
/* > \endverbatim */
/* > */
/* > \param[in,out] VF */
/* > \verbatim */
/* >          VF is DOUBLE PRECISION array, dimension ( K ) */
/* >          On entry, VF contains  information passed through DBEDE8. */
/* >          On exit, VF contains the first K components of the first */
/* >          components of all right singular vectors of the bidiagonal */
/* >          matrix. */
/* > \endverbatim */
/* > */
/* > \param[in,out] VL */
/* > \verbatim */
/* >          VL is DOUBLE PRECISION array, dimension ( K ) */
/* >          On entry, VL contains  information passed through DBEDE8. */
/* >          On exit, VL contains the first K components of the last */
/* >          components of all right singular vectors of the bidiagonal */
/* >          matrix. */
/* > \endverbatim */
/* > */
/* > \param[out] DIFL */
/* > \verbatim */
/* >          DIFL is DOUBLE PRECISION array, dimension ( K ) */
/* >          On exit, DIFL(I) = D(I) - DSIGMA(I). */
/* > \endverbatim */
/* > */
/* > \param[out] DIFR */
/* > \verbatim */
/* >          DIFR is DOUBLE PRECISION array, */
/* >                   dimension ( LDDIFR, 2 ) if ICOMPQ = 1 and */
/* >                   dimension ( K ) if ICOMPQ = 0. */
/* >          On exit, DIFR(I,1) = D(I) - DSIGMA(I+1), DIFR(K,1) is not */
/* >          defined and will not be referenced. */
/* > */
/* >          If ICOMPQ = 1, DIFR(1:K,2) is an array containing the */
/* >          normalizing factors for the right singular vector matrix. */
/* > \endverbatim */
/* > */
/* > \param[in] LDDIFR */
/* > \verbatim */
/* >          LDDIFR is INTEGER */
/* >          The leading dimension of DIFR, must be at least K. */
/* > \endverbatim */
/* > */
/* > \param[in,out] DSIGMA */
/* > \verbatim */
/* >          DSIGMA is DOUBLE PRECISION array, dimension ( K ) */
/* >          On entry, the first K elements of this array contain the old */
/* >          roots of the deflated updating problem.  These are the poles */
/* >          of the secular equation. */
/* >          On exit, the elements of DSIGMA may be very slightly altered */
/* >          in value. */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is DOUBLE PRECISION array, dimension (3*K) */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit. */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/* >          > 0:  if INFO = 1, a singular value did not converge */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date June 2017 */

/* > \ingroup OTHERauxiliary */

/* > \par Contributors: */
/*  ================== */
/* > */
/* >     Ming Gu and Huan Ren, Computer Science Division, University of */
/* >     California at Berkeley, USA */
/* > */
/*  ===================================================================== */
void  hlasd8_(integer *icompq, integer *k, halfreal *d__, 
	halfreal *z__, halfreal *vf, halfreal *vl, halfreal *difl, 
	halfreal *difr, integer *lddifr, halfreal *dsigma, halfreal *
	work, integer *info)
{
    /* System generated locals */
    integer difr_dim1, difr_offset, i__1, i__2;
    halfreal d__1, d__2;

    /* Local variables */
    integer i__, j;
    halfreal dj, rho;
    integer iwk1, iwk2, iwk3;
    extern halfreal hdot_(integer *, halfreal *, integer *, halfreal *, 
	    integer *);
    halfreal temp;
    extern halfreal hnrm2_(integer *, halfreal *, integer *);
    integer iwk2i, iwk3i;
    halfreal diflj, difrj, dsigj;
    extern void  hcopy_(integer *, halfreal *, integer *, 
	    halfreal *, integer *);
    extern halfreal hlamc3_(halfreal *, halfreal *);
    extern void  hlasd4_(integer *, integer *, halfreal *, 
	    halfreal *, halfreal *, halfreal *, halfreal *, 
	    halfreal *, integer *), hlascl_(char *, integer *, integer *, 
	    halfreal *, halfreal *, integer *, integer *, halfreal *, 
	    integer *, integer *), hlaset_(char *, integer *, integer 
	    *, halfreal *, halfreal *, halfreal *, integer *), 
	    xerbla_(char *, integer *);
    halfreal dsigjp;


/*  -- LAPACK auxiliary routine (version 3.7.1) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     June 2017 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    --d__;
    --z__;
    --vf;
    --vl;
    --difl;
    difr_dim1 = *lddifr;
    difr_offset = 1 + difr_dim1;
    difr -= difr_offset;
    --dsigma;
    --work;

    /* Function Body */
    *info = 0;

    if (*icompq < 0 || *icompq > 1) {
	*info = -1;
    } else if (*k < 1) {
	*info = -2;
    } else if (*lddifr < *k) {
	*info = -9;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DLASD8", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*k == 1) {
	d__[1] = abs(z__[1]);
	difl[1] = d__[1];
	if (*icompq == 1) {
	    difl[2] = 1.;
	    difr[(difr_dim1 << 1) + 1] = 1.;
	}
	return;
    }

/*     Modify values DSIGMA(i) to make sure all DSIGMA(i)-DSIGMA(j) can */
/*     be computed with high relative accuracy (barring over/underflow). */
/*     This is a problem on machines without a guard digit in */
/*     add/subtract (Cray XMP, Cray YMP, Cray C 90 and Cray 2). */
/*     The following code replaces DSIGMA(I) by 2*DSIGMA(I)-DSIGMA(I), */
/*     which on any of these machines zeros out the bottommost */
/*     bit of DSIGMA(I) if it is 1; this makes the subsequent */
/*     subtractions DSIGMA(I)-DSIGMA(J) unproblematic when cancellation */
/*     occurs. On binary machines with a guard digit (almost all */
/*     machines) it does not change DSIGMA(I) at all. On hexadecimal */
/*     and decimal machines with a guard digit, it slightly */
/*     changes the bottommost bits of DSIGMA(I). It does not account */
/*     for hexadecimal or decimal machines without guard digits */
/*     (we know of none). We use a subroutine call to compute */
/*     2*DLAMBDA(I) to prevent optimizing compilers from eliminating */
/*     this code. */

    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dsigma[i__] = hlamc3_(&dsigma[i__], &dsigma[i__]) - dsigma[i__];
/* L10: */
    }

/*     Book keeping. */

    iwk1 = 1;
    iwk2 = iwk1 + *k;
    iwk3 = iwk2 + *k;
    iwk2i = iwk2 - 1;
    iwk3i = iwk3 - 1;

/*     Normalize Z. */

    rho = hnrm2_(k, &z__[1], &c__1);
    hlascl_("G", &c__0, &c__0, &rho, &c_b8, k, &c__1, &z__[1], k, info);
    rho *= rho;

/*     Initialize WORK(IWK3). */

    hlaset_("A", k, &c__1, &c_b8, &c_b8, &work[iwk3], k);

/*     Compute the updated singular values, the arrays DIFL, DIFR, */
/*     and the updated Z. */

    i__1 = *k;
    for (j = 1; j <= i__1; ++j) {
	hlasd4_(k, &j, &dsigma[1], &z__[1], &work[iwk1], &rho, &d__[j], &work[
		iwk2], info);

/*        If the root finder fails, report the convergence failure. */

	if (*info != 0) {
	    return;
	}
	work[iwk3i + j] = work[iwk3i + j] * work[j] * work[iwk2i + j];
	difl[j] = -work[j];
	difr[j + difr_dim1] = -work[j + 1];
	i__2 = j - 1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    work[iwk3i + i__] = work[iwk3i + i__] * work[i__] * work[iwk2i + 
		    i__] / (dsigma[i__] - dsigma[j]) / (dsigma[i__] + dsigma[
		    j]);
/* L20: */
	}
	i__2 = *k;
	for (i__ = j + 1; i__ <= i__2; ++i__) {
	    work[iwk3i + i__] = work[iwk3i + i__] * work[i__] * work[iwk2i + 
		    i__] / (dsigma[i__] - dsigma[j]) / (dsigma[i__] + dsigma[
		    j]);
/* L30: */
	}
/* L40: */
    }

/*     Compute updated Z. */

    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	d__2 = M(sqrt)((d__1 = work[iwk3i + i__], abs(d__1)));
	z__[i__] = d_sign(&d__2, &z__[i__]);
/* L50: */
    }

/*     Update VF and VL. */

    i__1 = *k;
    for (j = 1; j <= i__1; ++j) {
	diflj = difl[j];
	dj = d__[j];
	dsigj = -dsigma[j];
	if (j < *k) {
	    difrj = -difr[j + difr_dim1];
	    dsigjp = -dsigma[j + 1];
	}
	work[j] = -z__[j] / diflj / (dsigma[j] + dj);
	i__2 = j - 1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    work[i__] = z__[i__] / (hlamc3_(&dsigma[i__], &dsigj) - diflj) / (
		    dsigma[i__] + dj);
/* L60: */
	}
	i__2 = *k;
	for (i__ = j + 1; i__ <= i__2; ++i__) {
	    work[i__] = z__[i__] / (hlamc3_(&dsigma[i__], &dsigjp) + difrj) / 
		    (dsigma[i__] + dj);
/* L70: */
	}
	temp = hnrm2_(k, &work[1], &c__1);
	work[iwk2i + j] = hdot_(k, &work[1], &c__1, &vf[1], &c__1) / temp;
	work[iwk3i + j] = hdot_(k, &work[1], &c__1, &vl[1], &c__1) / temp;
	if (*icompq == 1) {
	    difr[j + (difr_dim1 << 1)] = temp;
	}
/* L80: */
    }

    hcopy_(k, &work[iwk2], &c__1, &vf[1], &c__1);
    hcopy_(k, &work[iwk3], &c__1, &vl[1], &c__1);

    return;

/*     End of DLASD8 */

} /* hlasd8_ */

