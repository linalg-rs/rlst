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
static halfreal c_b5 = -1.;

/* > \brief \b DORBDB4 */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DORBDB4 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dorbdb4
.f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dorbdb4
.f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dorbdb4
.f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DORBDB4( M, P, Q, X11, LDX11, X21, LDX21, THETA, PHI, */
/*                           TAUP1, TAUP2, TAUQ1, PHANTOM, WORK, LWORK, */
/*                           INFO ) */

/*       INTEGER            INFO, LWORK, M, P, Q, LDX11, LDX21 */
/*       DOUBLE PRECISION   PHI(*), THETA(*) */
/*       DOUBLE PRECISION   PHANTOM(*), TAUP1(*), TAUP2(*), TAUQ1(*), */
/*      $                   WORK(*), X11(LDX11,*), X21(LDX21,*) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* >\verbatim */
/* > */
/* > DORBDB4 simultaneously bidiagonalizes the blocks of a tall and skinny */
/* > matrix X with orthonomal columns: */
/* > */
/* >                            [ B11 ] */
/* >      [ X11 ]   [ P1 |    ] [  0  ] */
/* >      [-----] = [---------] [-----] Q1**T . */
/* >      [ X21 ]   [    | P2 ] [ B21 ] */
/* >                            [  0  ] */
/* > */
/* > X11 is P-by-Q, and X21 is (M-P)-by-Q. M-Q must be no larger than P, */
/* > M-P, or Q. Routines DORBDB1, DORBDB2, and DORBDB3 handle cases in */
/* > which M-Q is not the minimum dimension. */
/* > */
/* > The orthogonal matrices P1, P2, and Q1 are P-by-P, (M-P)-by-(M-P), */
/* > and (M-Q)-by-(M-Q), respectively. They are represented implicitly by */
/* > Householder vectors. */
/* > */
/* > B11 and B12 are (M-Q)-by-(M-Q) bidiagonal matrices represented */
/* > implicitly by angles THETA, PHI. */
/* > */
/* >\endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >           The number of rows X11 plus the number of rows in X21. */
/* > \endverbatim */
/* > */
/* > \param[in] P */
/* > \verbatim */
/* >          P is INTEGER */
/* >           The number of rows in X11. 0 <= P <= M. */
/* > \endverbatim */
/* > */
/* > \param[in] Q */
/* > \verbatim */
/* >          Q is INTEGER */
/* >           The number of columns in X11 and X21. 0 <= Q <= M and */
/* >           M-Q <= f2cmin(P,M-P,Q). */
/* > \endverbatim */
/* > */
/* > \param[in,out] X11 */
/* > \verbatim */
/* >          X11 is DOUBLE PRECISION array, dimension (LDX11,Q) */
/* >           On entry, the top block of the matrix X to be reduced. On */
/* >           exit, the columns of tril(X11) specify reflectors for P1 and */
/* >           the rows of triu(X11,1) specify reflectors for Q1. */
/* > \endverbatim */
/* > */
/* > \param[in] LDX11 */
/* > \verbatim */
/* >          LDX11 is INTEGER */
/* >           The leading dimension of X11. LDX11 >= P. */
/* > \endverbatim */
/* > */
/* > \param[in,out] X21 */
/* > \verbatim */
/* >          X21 is DOUBLE PRECISION array, dimension (LDX21,Q) */
/* >           On entry, the bottom block of the matrix X to be reduced. On */
/* >           exit, the columns of tril(X21) specify reflectors for P2. */
/* > \endverbatim */
/* > */
/* > \param[in] LDX21 */
/* > \verbatim */
/* >          LDX21 is INTEGER */
/* >           The leading dimension of X21. LDX21 >= M-P. */
/* > \endverbatim */
/* > */
/* > \param[out] THETA */
/* > \verbatim */
/* >          THETA is DOUBLE PRECISION array, dimension (Q) */
/* >           The entries of the bidiagonal blocks B11, B21 are defined by */
/* >           THETA and PHI. See Further Details. */
/* > \endverbatim */
/* > */
/* > \param[out] PHI */
/* > \verbatim */
/* >          PHI is DOUBLE PRECISION array, dimension (Q-1) */
/* >           The entries of the bidiagonal blocks B11, B21 are defined by */
/* >           THETA and PHI. See Further Details. */
/* > \endverbatim */
/* > */
/* > \param[out] TAUP1 */
/* > \verbatim */
/* >          TAUP1 is DOUBLE PRECISION array, dimension (P) */
/* >           The scalar factors of the elementary reflectors that define */
/* >           P1. */
/* > \endverbatim */
/* > */
/* > \param[out] TAUP2 */
/* > \verbatim */
/* >          TAUP2 is DOUBLE PRECISION array, dimension (M-P) */
/* >           The scalar factors of the elementary reflectors that define */
/* >           P2. */
/* > \endverbatim */
/* > */
/* > \param[out] TAUQ1 */
/* > \verbatim */
/* >          TAUQ1 is DOUBLE PRECISION array, dimension (Q) */
/* >           The scalar factors of the elementary reflectors that define */
/* >           Q1. */
/* > \endverbatim */
/* > */
/* > \param[out] PHANTOM */
/* > \verbatim */
/* >          PHANTOM is DOUBLE PRECISION array, dimension (M) */
/* >           The routine computes an M-by-1 column vector Y that is */
/* >           orthogonal to the columns of [ X11; X21 ]. PHANTOM(1:P) and */
/* >           PHANTOM(P+1:M) contain Householder vectors for Y(1:P) and */
/* >           Y(P+1:M), respectively. */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is DOUBLE PRECISION array, dimension (LWORK) */
/* > \endverbatim */
/* > */
/* > \param[in] LWORK */
/* > \verbatim */
/* >          LWORK is INTEGER */
/* >           The dimension of the array WORK. LWORK >= M-Q. */
/* > */
/* >           If LWORK = -1, then a workspace query is assumed; the routine */
/* >           only calculates the optimal size of the WORK array, returns */
/* >           this value as the first entry of the WORK array, and no error */
/* >           message related to LWORK is issued by XERBLA. */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >           = 0:  successful exit. */
/* >           < 0:  if INFO = -i, the i-th argument had an illegal value. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date July 2012 */

/* > \ingroup doubleOTHERcomputational */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > */
/* >  The upper-bidiagonal blocks B11, B21 are represented implicitly by */
/* >  angles THETA(1), ..., THETA(Q) and PHI(1), ..., PHI(Q-1). Every entry */
/* >  in each bidiagonal band is a product of a sine or cosine of a THETA */
/* >  with a sine or cosine of a PHI. See [1] or DORCSD for details. */
/* > */
/* >  P1, P2, and Q1 are represented as products of elementary reflectors. */
/* >  See DORCSD2BY1 for details on generating P1, P2, and Q1 using DORGQR */
/* >  and DORGLQ. */
/* > \endverbatim */

/* > \par References: */
/*  ================ */
/* > */
/* >  [1] Brian D. Sutton. Computing the complete CS decomposition. Numer. */
/* >      Algorithms, 50(1):33-65, 2009. */
/* > */
/*  ===================================================================== */
void  horbdb4_(integer *m, integer *p, integer *q, halfreal *
	x11, integer *ldx11, halfreal *x21, integer *ldx21, halfreal *
	theta, halfreal *phi, halfreal *taup1, halfreal *taup2, 
	halfreal *tauq1, halfreal *phantom, halfreal *work, integer *
	lwork, integer *info)
{
    /* System generated locals */
    integer x11_dim1, x11_offset, x21_dim1, x21_offset, i__1, i__2, i__3, 
	    i__4;
    halfreal d__1, d__2;

    /* Local variables */
    integer lworkmin, lworkopt;
    halfreal c__;
    integer i__, j;
    halfreal s;
    integer childinfo;
    extern void  hrot_(integer *, halfreal *, integer *, 
	    halfreal *, integer *, halfreal *, halfreal *);
    extern halfreal hnrm2_(integer *, halfreal *, integer *);
    extern void  hscal_(integer *, halfreal *, halfreal *, 
	    integer *), hlarf_(char *, integer *, integer *, halfreal *, 
	    integer *, halfreal *, halfreal *, integer *, halfreal *);
    integer ilarf, llarf;
    extern void  xerbla_(char *, integer *);
    logical lquery;
    extern void  horbdb5_(integer *, integer *, integer *, 
	    halfreal *, integer *, halfreal *, integer *, halfreal *, 
	    integer *, halfreal *, integer *, halfreal *, integer *, 
	    integer *);
    integer iorbdb5, lorbdb5;
    extern void  hlarfgp_(integer *, halfreal *, halfreal *
	    , integer *, halfreal *);


/*  -- LAPACK computational routine (version 3.7.1) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     July 2012 */


/*  ==================================================================== */


/*     Test input arguments */

    /* Parameter adjustments */
    x11_dim1 = *ldx11;
    x11_offset = 1 + x11_dim1;
    x11 -= x11_offset;
    x21_dim1 = *ldx21;
    x21_offset = 1 + x21_dim1;
    x21 -= x21_offset;
    --theta;
    --phi;
    --taup1;
    --taup2;
    --tauq1;
    --phantom;
    --work;

    /* Function Body */
    *info = 0;
    lquery = *lwork == -1;

    if (*m < 0) {
	*info = -1;
    } else if (*p < *m - *q || *m - *p < *m - *q) {
	*info = -2;
    } else if (*q < *m - *q || *q > *m) {
	*info = -3;
    } else if (*ldx11 < f2cmax(1,*p)) {
	*info = -5;
    } else /* if(complicated condition) */ {
/* Computing MAX */
	i__1 = 1, i__2 = *m - *p;
	if (*ldx21 < f2cmax(i__1,i__2)) {
	    *info = -7;
	}
    }

/*     Compute workspace */

    if (*info == 0) {
	ilarf = 2;
/* Computing MAX */
	i__1 = *q - 1, i__2 = *p - 1, i__1 = f2cmax(i__1,i__2), i__2 = *m - *p - 
		1;
	llarf = f2cmax(i__1,i__2);
	iorbdb5 = 2;
	lorbdb5 = *q;
	lworkopt = ilarf + llarf - 1;
/* Computing MAX */
	i__1 = lworkopt, i__2 = iorbdb5 + lorbdb5 - 1;
	lworkopt = f2cmax(i__1,i__2);
	lworkmin = lworkopt;
	work[1] = (halfreal) lworkopt;
	if (*lwork < lworkmin && ! lquery) {
	    *info = -14;
	}
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DORBDB4", &i__1);
	return;
    } else if (lquery) {
	return;
    }

/*     Reduce columns 1, ..., M-Q of X11 and X21 */

    i__1 = *m - *q;
    for (i__ = 1; i__ <= i__1; ++i__) {

	if (i__ == 1) {
	    i__2 = *m;
	    for (j = 1; j <= i__2; ++j) {
		phantom[j] = 0.;
	    }
	    i__2 = *m - *p;
	    horbdb5_(p, &i__2, q, &phantom[1], &c__1, &phantom[*p + 1], &c__1,
		     &x11[x11_offset], ldx11, &x21[x21_offset], ldx21, &work[
		    iorbdb5], &lorbdb5, &childinfo);
	    hscal_(p, &c_b5, &phantom[1], &c__1);
	    hlarfgp_(p, &phantom[1], &phantom[2], &c__1, &taup1[1]);
	    i__2 = *m - *p;
	    hlarfgp_(&i__2, &phantom[*p + 1], &phantom[*p + 2], &c__1, &taup2[
		    1]);
	    theta[i__] = atan2(phantom[1], phantom[*p + 1]);
	    c__ = M(cos)(theta[i__]);
	    s = M(sin)(theta[i__]);
	    phantom[1] = 1.;
	    phantom[*p + 1] = 1.;
	    hlarf_("L", p, q, &phantom[1], &c__1, &taup1[1], &x11[x11_offset],
		     ldx11, &work[ilarf]);
	    i__2 = *m - *p;
	    hlarf_("L", &i__2, q, &phantom[*p + 1], &c__1, &taup2[1], &x21[
		    x21_offset], ldx21, &work[ilarf]);
	} else {
	    i__2 = *p - i__ + 1;
	    i__3 = *m - *p - i__ + 1;
	    i__4 = *q - i__ + 1;
	    horbdb5_(&i__2, &i__3, &i__4, &x11[i__ + (i__ - 1) * x11_dim1], &
		    c__1, &x21[i__ + (i__ - 1) * x21_dim1], &c__1, &x11[i__ + 
		    i__ * x11_dim1], ldx11, &x21[i__ + i__ * x21_dim1], ldx21,
		     &work[iorbdb5], &lorbdb5, &childinfo);
	    i__2 = *p - i__ + 1;
	    hscal_(&i__2, &c_b5, &x11[i__ + (i__ - 1) * x11_dim1], &c__1);
	    i__2 = *p - i__ + 1;
	    hlarfgp_(&i__2, &x11[i__ + (i__ - 1) * x11_dim1], &x11[i__ + 1 + (
		    i__ - 1) * x11_dim1], &c__1, &taup1[i__]);
	    i__2 = *m - *p - i__ + 1;
	    hlarfgp_(&i__2, &x21[i__ + (i__ - 1) * x21_dim1], &x21[i__ + 1 + (
		    i__ - 1) * x21_dim1], &c__1, &taup2[i__]);
	    theta[i__] = atan2(x11[i__ + (i__ - 1) * x11_dim1], x21[i__ + (
		    i__ - 1) * x21_dim1]);
	    c__ = M(cos)(theta[i__]);
	    s = M(sin)(theta[i__]);
	    x11[i__ + (i__ - 1) * x11_dim1] = 1.;
	    x21[i__ + (i__ - 1) * x21_dim1] = 1.;
	    i__2 = *p - i__ + 1;
	    i__3 = *q - i__ + 1;
	    hlarf_("L", &i__2, &i__3, &x11[i__ + (i__ - 1) * x11_dim1], &c__1,
		     &taup1[i__], &x11[i__ + i__ * x11_dim1], ldx11, &work[
		    ilarf]);
	    i__2 = *m - *p - i__ + 1;
	    i__3 = *q - i__ + 1;
	    hlarf_("L", &i__2, &i__3, &x21[i__ + (i__ - 1) * x21_dim1], &c__1,
		     &taup2[i__], &x21[i__ + i__ * x21_dim1], ldx21, &work[
		    ilarf]);
	}

	i__2 = *q - i__ + 1;
	d__1 = -c__;
	hrot_(&i__2, &x11[i__ + i__ * x11_dim1], ldx11, &x21[i__ + i__ * 
		x21_dim1], ldx21, &s, &d__1);
	i__2 = *q - i__ + 1;
	hlarfgp_(&i__2, &x21[i__ + i__ * x21_dim1], &x21[i__ + (i__ + 1) * 
		x21_dim1], ldx21, &tauq1[i__]);
	c__ = x21[i__ + i__ * x21_dim1];
	x21[i__ + i__ * x21_dim1] = 1.;
	i__2 = *p - i__;
	i__3 = *q - i__ + 1;
	hlarf_("R", &i__2, &i__3, &x21[i__ + i__ * x21_dim1], ldx21, &tauq1[
		i__], &x11[i__ + 1 + i__ * x11_dim1], ldx11, &work[ilarf]);
	i__2 = *m - *p - i__;
	i__3 = *q - i__ + 1;
	hlarf_("R", &i__2, &i__3, &x21[i__ + i__ * x21_dim1], ldx21, &tauq1[
		i__], &x21[i__ + 1 + i__ * x21_dim1], ldx21, &work[ilarf]);
	if (i__ < *m - *q) {
	    i__2 = *p - i__;
/* Computing 2nd power */
	    d__1 = hnrm2_(&i__2, &x11[i__ + 1 + i__ * x11_dim1], &c__1);
	    i__3 = *m - *p - i__;
/* Computing 2nd power */
	    d__2 = hnrm2_(&i__3, &x21[i__ + 1 + i__ * x21_dim1], &c__1);
	    s = M(sqrt)(d__1 * d__1 + d__2 * d__2);
	    phi[i__] = atan2(s, c__);
	}

    }

/*     Reduce the bottom-right portion of X11 to [ I 0 ] */

    i__1 = *p;
    for (i__ = *m - *q + 1; i__ <= i__1; ++i__) {
	i__2 = *q - i__ + 1;
	hlarfgp_(&i__2, &x11[i__ + i__ * x11_dim1], &x11[i__ + (i__ + 1) * 
		x11_dim1], ldx11, &tauq1[i__]);
	x11[i__ + i__ * x11_dim1] = 1.;
	i__2 = *p - i__;
	i__3 = *q - i__ + 1;
	hlarf_("R", &i__2, &i__3, &x11[i__ + i__ * x11_dim1], ldx11, &tauq1[
		i__], &x11[i__ + 1 + i__ * x11_dim1], ldx11, &work[ilarf]);
	i__2 = *q - *p;
	i__3 = *q - i__ + 1;
	hlarf_("R", &i__2, &i__3, &x11[i__ + i__ * x11_dim1], ldx11, &tauq1[
		i__], &x21[*m - *q + 1 + i__ * x21_dim1], ldx21, &work[ilarf]);
    }

/*     Reduce the bottom-right portion of X21 to [ 0 I ] */

    i__1 = *q;
    for (i__ = *p + 1; i__ <= i__1; ++i__) {
	i__2 = *q - i__ + 1;
	hlarfgp_(&i__2, &x21[*m - *q + i__ - *p + i__ * x21_dim1], &x21[*m - *
		q + i__ - *p + (i__ + 1) * x21_dim1], ldx21, &tauq1[i__]);
	x21[*m - *q + i__ - *p + i__ * x21_dim1] = 1.;
	i__2 = *q - i__;
	i__3 = *q - i__ + 1;
	hlarf_("R", &i__2, &i__3, &x21[*m - *q + i__ - *p + i__ * x21_dim1], 
		ldx21, &tauq1[i__], &x21[*m - *q + i__ - *p + 1 + i__ * 
		x21_dim1], ldx21, &work[ilarf]);
    }

    return;

/*     End of DORBDB4 */

} /* horbdb4_ */

