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
static integer c__0 = 0;
static quadreal c_b13 = 1.;
static quadreal c_b26 = 0.;

/* > \brief \b DLASD3 finds all square roots of the roots of the secular equation, as defined by the values in
 D and Z, and then updates the singular vectors by matrix multiplication. Used by sbdsdc. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DLASD3 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasd3.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasd3.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasd3.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DLASD3( NL, NR, SQRE, K, D, Q, LDQ, DSIGMA, U, LDU, U2, */
/*                          LDU2, VT, LDVT, VT2, LDVT2, IDXC, CTOT, Z, */
/*                          INFO ) */

/*       INTEGER            INFO, K, LDQ, LDU, LDU2, LDVT, LDVT2, NL, NR, */
/*      $                   SQRE */
/*       INTEGER            CTOT( * ), IDXC( * ) */
/*       DOUBLE PRECISION   D( * ), DSIGMA( * ), Q( LDQ, * ), U( LDU, * ), */
/*      $                   U2( LDU2, * ), VT( LDVT, * ), VT2( LDVT2, * ), */
/*      $                   Z( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DLASD3 finds all the square roots of the roots of the secular */
/* > equation, as defined by the values in D and Z.  It makes the */
/* > appropriate calls to DLASD4 and then updates the singular */
/* > vectors by matrix multiplication. */
/* > */
/* > This code makes very mild assumptions about floating point */
/* > arithmetic. It will work on machines with a guard digit in */
/* > add/subtract, or on those binary machines without guard digits */
/* > which subtract like the Cray XMP, Cray YMP, Cray C 90, or Cray 2. */
/* > It could conceivably fail on hexadecimal or decimal machines */
/* > without guard digits, but we know of none. */
/* > */
/* > DLASD3 is called from DLASD1. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] NL */
/* > \verbatim */
/* >          NL is INTEGER */
/* >         The row dimension of the upper block.  NL >= 1. */
/* > \endverbatim */
/* > */
/* > \param[in] NR */
/* > \verbatim */
/* >          NR is INTEGER */
/* >         The row dimension of the lower block.  NR >= 1. */
/* > \endverbatim */
/* > */
/* > \param[in] SQRE */
/* > \verbatim */
/* >          SQRE is INTEGER */
/* >         = 0: the lower block is an NR-by-NR square matrix. */
/* >         = 1: the lower block is an NR-by-(NR+1) rectangular matrix. */
/* > */
/* >         The bidiagonal matrix has N = NL + NR + 1 rows and */
/* >         M = N + SQRE >= N columns. */
/* > \endverbatim */
/* > */
/* > \param[in] K */
/* > \verbatim */
/* >          K is INTEGER */
/* >         The size of the secular equation, 1 =< K = < N. */
/* > \endverbatim */
/* > */
/* > \param[out] D */
/* > \verbatim */
/* >          D is DOUBLE PRECISION array, dimension(K) */
/* >         On exit the square roots of the roots of the secular equation, */
/* >         in ascending order. */
/* > \endverbatim */
/* > */
/* > \param[out] Q */
/* > \verbatim */
/* >          Q is DOUBLE PRECISION array, dimension (LDQ,K) */
/* > \endverbatim */
/* > */
/* > \param[in] LDQ */
/* > \verbatim */
/* >          LDQ is INTEGER */
/* >         The leading dimension of the array Q.  LDQ >= K. */
/* > \endverbatim */
/* > */
/* > \param[in,out] DSIGMA */
/* > \verbatim */
/* >          DSIGMA is DOUBLE PRECISION array, dimension(K) */
/* >         The first K elements of this array contain the old roots */
/* >         of the deflated updating problem.  These are the poles */
/* >         of the secular equation. */
/* > \endverbatim */
/* > */
/* > \param[out] U */
/* > \verbatim */
/* >          U is DOUBLE PRECISION array, dimension (LDU, N) */
/* >         The last N - K columns of this matrix contain the deflated */
/* >         left singular vectors. */
/* > \endverbatim */
/* > */
/* > \param[in] LDU */
/* > \verbatim */
/* >          LDU is INTEGER */
/* >         The leading dimension of the array U.  LDU >= N. */
/* > \endverbatim */
/* > */
/* > \param[in] U2 */
/* > \verbatim */
/* >          U2 is DOUBLE PRECISION array, dimension (LDU2, N) */
/* >         The first K columns of this matrix contain the non-deflated */
/* >         left singular vectors for the split problem. */
/* > \endverbatim */
/* > */
/* > \param[in] LDU2 */
/* > \verbatim */
/* >          LDU2 is INTEGER */
/* >         The leading dimension of the array U2.  LDU2 >= N. */
/* > \endverbatim */
/* > */
/* > \param[out] VT */
/* > \verbatim */
/* >          VT is DOUBLE PRECISION array, dimension (LDVT, M) */
/* >         The last M - K columns of VT**T contain the deflated */
/* >         right singular vectors. */
/* > \endverbatim */
/* > */
/* > \param[in] LDVT */
/* > \verbatim */
/* >          LDVT is INTEGER */
/* >         The leading dimension of the array VT.  LDVT >= N. */
/* > \endverbatim */
/* > */
/* > \param[in,out] VT2 */
/* > \verbatim */
/* >          VT2 is DOUBLE PRECISION array, dimension (LDVT2, N) */
/* >         The first K columns of VT2**T contain the non-deflated */
/* >         right singular vectors for the split problem. */
/* > \endverbatim */
/* > */
/* > \param[in] LDVT2 */
/* > \verbatim */
/* >          LDVT2 is INTEGER */
/* >         The leading dimension of the array VT2.  LDVT2 >= N. */
/* > \endverbatim */
/* > */
/* > \param[in] IDXC */
/* > \verbatim */
/* >          IDXC is INTEGER array, dimension ( N ) */
/* >         The permutation used to arrange the columns of U (and rows of */
/* >         VT) into three groups:  the first group contains non-zero */
/* >         entries only at and above (or before) NL +1; the second */
/* >         contains non-zero entries only at and below (or after) NL+2; */
/* >         and the third is dense. The first column of U and the row of */
/* >         VT are treated separately, however. */
/* > */
/* >         The rows of the singular vectors found by DLASD4 */
/* >         must be likewise permuted before the matrix multiplies can */
/* >         take place. */
/* > \endverbatim */
/* > */
/* > \param[in] CTOT */
/* > \verbatim */
/* >          CTOT is INTEGER array, dimension ( 4 ) */
/* >         A count of the total number of the various types of columns */
/* >         in U (or rows in VT), as described in IDXC. The fourth column */
/* >         type is any column which has been deflated. */
/* > \endverbatim */
/* > */
/* > \param[in,out] Z */
/* > \verbatim */
/* >          Z is DOUBLE PRECISION array, dimension (K) */
/* >         The first K elements of this array contain the components */
/* >         of the deflation-adjusted updating row vector. */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >         = 0:  successful exit. */
/* >         < 0:  if INFO = -i, the i-th argument had an illegal value. */
/* >         > 0:  if INFO = 1, a singular value did not converge */
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
void  qlasd3_(integer *nl, integer *nr, integer *sqre, integer 
	*k, quadreal *d__, quadreal *q, integer *ldq, quadreal *dsigma, 
	quadreal *u, integer *ldu, quadreal *u2, integer *ldu2, 
	quadreal *vt, integer *ldvt, quadreal *vt2, integer *ldvt2, 
	integer *idxc, integer *ctot, quadreal *z__, integer *info)
{
    /* System generated locals */
    integer q_dim1, q_offset, u_dim1, u_offset, u2_dim1, u2_offset, vt_dim1, 
	    vt_offset, vt2_dim1, vt2_offset, i__1, i__2;
    quadreal d__1, d__2;

    /* Local variables */
    integer i__, j, m, n, jc;
    quadreal rho;
    integer nlp1, nlp2, nrp1;
    quadreal temp;
    extern quadreal qnrm2_(integer *, quadreal *, integer *);
    extern void  qgemm_(char *, char *, integer *, integer *, 
	    integer *, quadreal *, quadreal *, integer *, quadreal *, 
	    integer *, quadreal *, quadreal *, integer *);
    integer ctemp;
    extern void  qcopy_(integer *, quadreal *, integer *, 
	    quadreal *, integer *);
    integer ktemp;
    extern quadreal qlamc3_(quadreal *, quadreal *);
    extern void  qlasd4_(integer *, integer *, quadreal *, 
	    quadreal *, quadreal *, quadreal *, quadreal *, 
	    quadreal *, integer *), qlascl_(char *, integer *, integer *, 
	    quadreal *, quadreal *, integer *, integer *, quadreal *, 
	    integer *, integer *), qlacpy_(char *, integer *, integer 
	    *, quadreal *, integer *, quadreal *, integer *), 
	    xerbla_(char *, integer *);


/*  -- LAPACK auxiliary routine (version 3.7.1) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     June 2017 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    --d__;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --dsigma;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    u2_dim1 = *ldu2;
    u2_offset = 1 + u2_dim1;
    u2 -= u2_offset;
    vt_dim1 = *ldvt;
    vt_offset = 1 + vt_dim1;
    vt -= vt_offset;
    vt2_dim1 = *ldvt2;
    vt2_offset = 1 + vt2_dim1;
    vt2 -= vt2_offset;
    --idxc;
    --ctot;
    --z__;

    /* Function Body */
    *info = 0;

    if (*nl < 1) {
	*info = -1;
    } else if (*nr < 1) {
	*info = -2;
    } else if (*sqre != 1 && *sqre != 0) {
	*info = -3;
    }

    n = *nl + *nr + 1;
    m = n + *sqre;
    nlp1 = *nl + 1;
    nlp2 = *nl + 2;

    if (*k < 1 || *k > n) {
	*info = -4;
    } else if (*ldq < *k) {
	*info = -7;
    } else if (*ldu < n) {
	*info = -10;
    } else if (*ldu2 < n) {
	*info = -12;
    } else if (*ldvt < m) {
	*info = -14;
    } else if (*ldvt2 < m) {
	*info = -16;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DLASD3", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*k == 1) {
	d__[1] = abs(z__[1]);
	qcopy_(&m, &vt2[vt2_dim1 + 1], ldvt2, &vt[vt_dim1 + 1], ldvt);
	if (z__[1] > 0.) {
	    qcopy_(&n, &u2[u2_dim1 + 1], &c__1, &u[u_dim1 + 1], &c__1);
	} else {
	    i__1 = n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		u[i__ + u_dim1] = -u2[i__ + u2_dim1];
/* L10: */
	    }
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
/*     2*DSIGMA(I) to prevent optimizing compilers from eliminating */
/*     this code. */

    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dsigma[i__] = qlamc3_(&dsigma[i__], &dsigma[i__]) - dsigma[i__];
/* L20: */
    }

/*     Keep a copy of Z. */

    qcopy_(k, &z__[1], &c__1, &q[q_offset], &c__1);

/*     Normalize Z. */

    rho = qnrm2_(k, &z__[1], &c__1);
    qlascl_("G", &c__0, &c__0, &rho, &c_b13, k, &c__1, &z__[1], k, info);
    rho *= rho;

/*     Find the new singular values. */

    i__1 = *k;
    for (j = 1; j <= i__1; ++j) {
	qlasd4_(k, &j, &dsigma[1], &z__[1], &u[j * u_dim1 + 1], &rho, &d__[j],
		 &vt[j * vt_dim1 + 1], info);

/*        If the zero finder fails, report the convergence failure. */

	if (*info != 0) {
	    return;
	}
/* L30: */
    }

/*     Compute updated Z. */

    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	z__[i__] = u[i__ + *k * u_dim1] * vt[i__ + *k * vt_dim1];
	i__2 = i__ - 1;
	for (j = 1; j <= i__2; ++j) {
	    z__[i__] *= u[i__ + j * u_dim1] * vt[i__ + j * vt_dim1] / (dsigma[
		    i__] - dsigma[j]) / (dsigma[i__] + dsigma[j]);
/* L40: */
	}
	i__2 = *k - 1;
	for (j = i__; j <= i__2; ++j) {
	    z__[i__] *= u[i__ + j * u_dim1] * vt[i__ + j * vt_dim1] / (dsigma[
		    i__] - dsigma[j + 1]) / (dsigma[i__] + dsigma[j + 1]);
/* L50: */
	}
	d__2 = M(sqrt)((d__1 = z__[i__], abs(d__1)));
	z__[i__] = d_sign(&d__2, &q[i__ + q_dim1]);
/* L60: */
    }

/*     Compute left singular vectors of the modified diagonal matrix, */
/*     and store related information for the right singular vectors. */

    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	vt[i__ * vt_dim1 + 1] = z__[1] / u[i__ * u_dim1 + 1] / vt[i__ * 
		vt_dim1 + 1];
	u[i__ * u_dim1 + 1] = -1.;
	i__2 = *k;
	for (j = 2; j <= i__2; ++j) {
	    vt[j + i__ * vt_dim1] = z__[j] / u[j + i__ * u_dim1] / vt[j + i__ 
		    * vt_dim1];
	    u[j + i__ * u_dim1] = dsigma[j] * vt[j + i__ * vt_dim1];
/* L70: */
	}
	temp = qnrm2_(k, &u[i__ * u_dim1 + 1], &c__1);
	q[i__ * q_dim1 + 1] = u[i__ * u_dim1 + 1] / temp;
	i__2 = *k;
	for (j = 2; j <= i__2; ++j) {
	    jc = idxc[j];
	    q[j + i__ * q_dim1] = u[jc + i__ * u_dim1] / temp;
/* L80: */
	}
/* L90: */
    }

/*     Update the left singular vector matrix. */

    if (*k == 2) {
	qgemm_("N", "N", &n, k, k, &c_b13, &u2[u2_offset], ldu2, &q[q_offset],
		 ldq, &c_b26, &u[u_offset], ldu);
	goto L100;
    }
    if (ctot[1] > 0) {
	qgemm_("N", "N", nl, k, &ctot[1], &c_b13, &u2[(u2_dim1 << 1) + 1], 
		ldu2, &q[q_dim1 + 2], ldq, &c_b26, &u[u_dim1 + 1], ldu);
	if (ctot[3] > 0) {
	    ktemp = ctot[1] + 2 + ctot[2];
	    qgemm_("N", "N", nl, k, &ctot[3], &c_b13, &u2[ktemp * u2_dim1 + 1]
		    , ldu2, &q[ktemp + q_dim1], ldq, &c_b13, &u[u_dim1 + 1], 
		    ldu);
	}
    } else if (ctot[3] > 0) {
	ktemp = ctot[1] + 2 + ctot[2];
	qgemm_("N", "N", nl, k, &ctot[3], &c_b13, &u2[ktemp * u2_dim1 + 1], 
		ldu2, &q[ktemp + q_dim1], ldq, &c_b26, &u[u_dim1 + 1], ldu);
    } else {
	qlacpy_("F", nl, k, &u2[u2_offset], ldu2, &u[u_offset], ldu);
    }
    qcopy_(k, &q[q_dim1 + 1], ldq, &u[nlp1 + u_dim1], ldu);
    ktemp = ctot[1] + 2;
    ctemp = ctot[2] + ctot[3];
    qgemm_("N", "N", nr, k, &ctemp, &c_b13, &u2[nlp2 + ktemp * u2_dim1], ldu2,
	     &q[ktemp + q_dim1], ldq, &c_b26, &u[nlp2 + u_dim1], ldu);

/*     Generate the right singular vectors. */

L100:
    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	temp = qnrm2_(k, &vt[i__ * vt_dim1 + 1], &c__1);
	q[i__ + q_dim1] = vt[i__ * vt_dim1 + 1] / temp;
	i__2 = *k;
	for (j = 2; j <= i__2; ++j) {
	    jc = idxc[j];
	    q[i__ + j * q_dim1] = vt[jc + i__ * vt_dim1] / temp;
/* L110: */
	}
/* L120: */
    }

/*     Update the right singular vector matrix. */

    if (*k == 2) {
	qgemm_("N", "N", k, &m, k, &c_b13, &q[q_offset], ldq, &vt2[vt2_offset]
		, ldvt2, &c_b26, &vt[vt_offset], ldvt);
	return;
    }
    ktemp = ctot[1] + 1;
    qgemm_("N", "N", k, &nlp1, &ktemp, &c_b13, &q[q_dim1 + 1], ldq, &vt2[
	    vt2_dim1 + 1], ldvt2, &c_b26, &vt[vt_dim1 + 1], ldvt);
    ktemp = ctot[1] + 2 + ctot[2];
    if (ktemp <= *ldvt2) {
	qgemm_("N", "N", k, &nlp1, &ctot[3], &c_b13, &q[ktemp * q_dim1 + 1], 
		ldq, &vt2[ktemp + vt2_dim1], ldvt2, &c_b13, &vt[vt_dim1 + 1], 
		ldvt);
    }

    ktemp = ctot[1] + 1;
    nrp1 = *nr + *sqre;
    if (ktemp > 1) {
	i__1 = *k;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    q[i__ + ktemp * q_dim1] = q[i__ + q_dim1];
/* L130: */
	}
	i__1 = m;
	for (i__ = nlp2; i__ <= i__1; ++i__) {
	    vt2[ktemp + i__ * vt2_dim1] = vt2[i__ * vt2_dim1 + 1];
/* L140: */
	}
    }
    ctemp = ctot[2] + 1 + ctot[3];
    qgemm_("N", "N", k, &nrp1, &ctemp, &c_b13, &q[ktemp * q_dim1 + 1], ldq, &
	    vt2[ktemp + nlp2 * vt2_dim1], ldvt2, &c_b26, &vt[nlp2 * vt_dim1 + 
	    1], ldvt);

    return;

/*     End of DLASD3 */

} /* qlasd3_ */

