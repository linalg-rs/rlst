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

static halfcomplex c_b1 = {0.,0.};
static halfcomplex c_b2 = {1.,0.};
static integer c__3 = 3;
static integer c__1 = 1;
static integer c__2 = 2;

/* > \brief \b ZLAQR5 performs a single small-bulge multi-shift QR sweep. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZLAQR5 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zlaqr5.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zlaqr5.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zlaqr5.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZLAQR5( WANTT, WANTZ, KACC22, N, KTOP, KBOT, NSHFTS, S, */
/*                          H, LDH, ILOZ, IHIZ, Z, LDZ, V, LDV, U, LDU, NV, */
/*                          WV, LDWV, NH, WH, LDWH ) */

/*       INTEGER            IHIZ, ILOZ, KACC22, KBOT, KTOP, LDH, LDU, LDV, */
/*      $                   LDWH, LDWV, LDZ, N, NH, NSHFTS, NV */
/*       LOGICAL            WANTT, WANTZ */
/*       COMPLEX*16         H( LDH, * ), S( * ), U( LDU, * ), V( LDV, * ), */
/*      $                   WH( LDWH, * ), WV( LDWV, * ), Z( LDZ, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* >    ZLAQR5, called by ZLAQR0, performs a */
/* >    single small-bulge multi-shift QR sweep. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] WANTT */
/* > \verbatim */
/* >          WANTT is LOGICAL */
/* >             WANTT = .true. if the triangular Schur factor */
/* >             is being computed.  WANTT is set to .false. otherwise. */
/* > \endverbatim */
/* > */
/* > \param[in] WANTZ */
/* > \verbatim */
/* >          WANTZ is LOGICAL */
/* >             WANTZ = .true. if the unitary Schur factor is being */
/* >             computed.  WANTZ is set to .false. otherwise. */
/* > \endverbatim */
/* > */
/* > \param[in] KACC22 */
/* > \verbatim */
/* >          KACC22 is INTEGER with value 0, 1, or 2. */
/* >             Specifies the computation mode of far-from-diagonal */
/* >             orthogonal updates. */
/* >        = 0: ZLAQR5 does not accumulate reflections and does not */
/* >             use matrix-matrix multiply to update far-from-diagonal */
/* >             matrix entries. */
/* >        = 1: ZLAQR5 accumulates reflections and uses matrix-matrix */
/* >             multiply to update the far-from-diagonal matrix entries. */
/* >        = 2: ZLAQR5 accumulates reflections, uses matrix-matrix */
/* >             multiply to update the far-from-diagonal matrix entries, */
/* >             and takes advantage of 2-by-2 block structure during */
/* >             matrix multiplies. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >             N is the order of the Hessenberg matrix H upon which this */
/* >             subroutine operates. */
/* > \endverbatim */
/* > */
/* > \param[in] KTOP */
/* > \verbatim */
/* >          KTOP is INTEGER */
/* > \endverbatim */
/* > */
/* > \param[in] KBOT */
/* > \verbatim */
/* >          KBOT is INTEGER */
/* >             These are the first and last rows and columns of an */
/* >             isolated diagonal block upon which the QR sweep is to be */
/* >             applied. It is assumed without a check that */
/* >                       either KTOP = 1  or   H(KTOP,KTOP-1) = 0 */
/* >             and */
/* >                       either KBOT = N  or   H(KBOT+1,KBOT) = 0. */
/* > \endverbatim */
/* > */
/* > \param[in] NSHFTS */
/* > \verbatim */
/* >          NSHFTS is INTEGER */
/* >             NSHFTS gives the number of simultaneous shifts.  NSHFTS */
/* >             must be positive and even. */
/* > \endverbatim */
/* > */
/* > \param[in,out] S */
/* > \verbatim */
/* >          S is COMPLEX*16 array, dimension (NSHFTS) */
/* >             S contains the shifts of origin that define the multi- */
/* >             shift QR sweep.  On output S may be reordered. */
/* > \endverbatim */
/* > */
/* > \param[in,out] H */
/* > \verbatim */
/* >          H is COMPLEX*16 array, dimension (LDH,N) */
/* >             On input H contains a Hessenberg matrix.  On output a */
/* >             multi-shift QR sweep with shifts SR(J)+i*SI(J) is applied */
/* >             to the isolated diagonal block in rows and columns KTOP */
/* >             through KBOT. */
/* > \endverbatim */
/* > */
/* > \param[in] LDH */
/* > \verbatim */
/* >          LDH is INTEGER */
/* >             LDH is the leading dimension of H just as declared in the */
/* >             calling procedure.  LDH.GE.MAX(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in] ILOZ */
/* > \verbatim */
/* >          ILOZ is INTEGER */
/* > \endverbatim */
/* > */
/* > \param[in] IHIZ */
/* > \verbatim */
/* >          IHIZ is INTEGER */
/* >             Specify the rows of Z to which transformations must be */
/* >             applied if WANTZ is .TRUE.. 1 .LE. ILOZ .LE. IHIZ .LE. N */
/* > \endverbatim */
/* > */
/* > \param[in,out] Z */
/* > \verbatim */
/* >          Z is COMPLEX*16 array, dimension (LDZ,IHIZ) */
/* >             If WANTZ = .TRUE., then the QR Sweep unitary */
/* >             similarity transformation is accumulated into */
/* >             Z(ILOZ:IHIZ,ILOZ:IHIZ) from the right. */
/* >             If WANTZ = .FALSE., then Z is unreferenced. */
/* > \endverbatim */
/* > */
/* > \param[in] LDZ */
/* > \verbatim */
/* >          LDZ is INTEGER */
/* >             LDA is the leading dimension of Z just as declared in */
/* >             the calling procedure. LDZ.GE.N. */
/* > \endverbatim */
/* > */
/* > \param[out] V */
/* > \verbatim */
/* >          V is COMPLEX*16 array, dimension (LDV,NSHFTS/2) */
/* > \endverbatim */
/* > */
/* > \param[in] LDV */
/* > \verbatim */
/* >          LDV is INTEGER */
/* >             LDV is the leading dimension of V as declared in the */
/* >             calling procedure.  LDV.GE.3. */
/* > \endverbatim */
/* > */
/* > \param[out] U */
/* > \verbatim */
/* >          U is COMPLEX*16 array, dimension (LDU,3*NSHFTS-3) */
/* > \endverbatim */
/* > */
/* > \param[in] LDU */
/* > \verbatim */
/* >          LDU is INTEGER */
/* >             LDU is the leading dimension of U just as declared in the */
/* >             in the calling subroutine.  LDU.GE.3*NSHFTS-3. */
/* > \endverbatim */
/* > */
/* > \param[in] NH */
/* > \verbatim */
/* >          NH is INTEGER */
/* >             NH is the number of columns in array WH available for */
/* >             workspace. NH.GE.1. */
/* > \endverbatim */
/* > */
/* > \param[out] WH */
/* > \verbatim */
/* >          WH is COMPLEX*16 array, dimension (LDWH,NH) */
/* > \endverbatim */
/* > */
/* > \param[in] LDWH */
/* > \verbatim */
/* >          LDWH is INTEGER */
/* >             Leading dimension of WH just as declared in the */
/* >             calling procedure.  LDWH.GE.3*NSHFTS-3. */
/* > \endverbatim */
/* > */
/* > \param[in] NV */
/* > \verbatim */
/* >          NV is INTEGER */
/* >             NV is the number of rows in WV agailable for workspace. */
/* >             NV.GE.1. */
/* > \endverbatim */
/* > */
/* > \param[out] WV */
/* > \verbatim */
/* >          WV is COMPLEX*16 array, dimension (LDWV,3*NSHFTS-3) */
/* > \endverbatim */
/* > */
/* > \param[in] LDWV */
/* > \verbatim */
/* >          LDWV is INTEGER */
/* >             LDWV is the leading dimension of WV as declared in the */
/* >             in the calling subroutine.  LDWV.GE.NV. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date June 2016 */

/* > \ingroup complex16OTHERauxiliary */

/* > \par Contributors: */
/*  ================== */
/* > */
/* >       Karen Braman and Ralph Byers, Department of Mathematics, */
/* >       University of Kansas, USA */

/* > \par References: */
/*  ================ */
/* > */
/* >       K. Braman, R. Byers and R. Mathias, The Multi-Shift QR */
/* >       Algorithm Part I: Maintaining Well Focused Shifts, and Level 3 */
/* >       Performance, SIAM Journal of Matrix Analysis, volume 23, pages */
/* >       929--947, 2002. */
/* > */
/*  ===================================================================== */
void  klaqr5_(logical *wantt, logical *wantz, integer *kacc22, 
	integer *n, integer *ktop, integer *kbot, integer *nshfts, 
	halfcomplex *s, halfcomplex *h__, integer *ldh, integer *iloz, 
	integer *ihiz, halfcomplex *z__, integer *ldz, halfcomplex *v, 
	integer *ldv, halfcomplex *u, integer *ldu, integer *nv, 
	halfcomplex *wv, integer *ldwv, integer *nh, halfcomplex *wh, 
	integer *ldwh)
{
    /* System generated locals */
    integer h_dim1, h_offset, u_dim1, u_offset, v_dim1, v_offset, wh_dim1, 
	    wh_offset, wv_dim1, wv_offset, z_dim1, z_offset, i__1, i__2, i__3,
	     i__4, i__5, i__6, i__7, i__8, i__9, i__10, i__11;
    halfreal d__1, d__2, d__3, d__4, d__5, d__6, d__7, d__8, d__9, d__10;
    halfcomplex z__1, z__2, z__3, z__4, z__5, z__6, z__7, z__8;

    /* Local variables */
    integer j, k, m, i2, j2, i4, j4, k1;
    halfreal h11, h12, h21, h22;
    integer m22, ns, nu;
    halfcomplex vt[3];
    halfreal scl;
    integer kdu, kms;
    halfreal ulp;
    integer knz, kzs;
    halfreal tst1, tst2;
    halfcomplex beta;
    logical blk22, bmp22;
    integer mend, jcol, jlen, jbot, mbot, jtop, jrow, mtop;
    halfcomplex alpha;
    logical accum;
    integer ndcol, incol, krcol, nbmps;
    extern void  kgemm_(char *, char *, integer *, integer *, 
	    integer *, halfcomplex *, halfcomplex *, integer *, 
	    halfcomplex *, integer *, halfcomplex *, halfcomplex *, 
	    integer *), ktrmm_(char *, char *, char *, char *,
	     integer *, integer *, halfcomplex *, halfcomplex *, integer *
	    , halfcomplex *, integer *), 
	    hlabad_(halfreal *, halfreal *), klaqr1_(integer *, 
	    halfcomplex *, integer *, halfcomplex *, halfcomplex *, 
	    halfcomplex *);
    extern halfreal hlamch_(char *);
    halfreal safmin, safmax;
    extern void  klarfg_(integer *, halfcomplex *, 
	    halfcomplex *, integer *, halfcomplex *);
    halfcomplex refsum;
    extern void  klacpy_(char *, integer *, integer *, 
	    halfcomplex *, integer *, halfcomplex *, integer *), 
	    klaset_(char *, integer *, integer *, halfcomplex *, 
	    halfcomplex *, halfcomplex *, integer *);
    integer mstart;
    halfreal smlnum;


/*  -- LAPACK auxiliary routine (version 3.7.1) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     June 2016 */


/*  ================================================================ */


/*     ==== If there are no shifts, then there is nothing to do. ==== */

    /* Parameter adjustments */
    --s;
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    wv_dim1 = *ldwv;
    wv_offset = 1 + wv_dim1;
    wv -= wv_offset;
    wh_dim1 = *ldwh;
    wh_offset = 1 + wh_dim1;
    wh -= wh_offset;

    /* Function Body */
    if (*nshfts < 2) {
	return;
    }

/*     ==== If the active block is empty or 1-by-1, then there */
/*     .    is nothing to do. ==== */

    if (*ktop >= *kbot) {
	return;
    }

/*     ==== NSHFTS is supposed to be even, but if it is odd, */
/*     .    then simply reduce it by one.  ==== */

    ns = *nshfts - *nshfts % 2;

/*     ==== Machine constants for deflation ==== */

    safmin = hlamch_("SAFE MINIMUM");
    safmax = 1. / safmin;
    hlabad_(&safmin, &safmax);
    ulp = hlamch_("PRECISION");
    smlnum = safmin * ((halfreal) (*n) / ulp);

/*     ==== Use accumulated reflections to update far-from-diagonal */
/*     .    entries ? ==== */

    accum = *kacc22 == 1 || *kacc22 == 2;

/*     ==== If so, exploit the 2-by-2 block structure? ==== */

    blk22 = ns > 2 && *kacc22 == 2;

/*     ==== clear trash ==== */

    if (*ktop + 2 <= *kbot) {
	i__1 = *ktop + 2 + *ktop * h_dim1;
	h__[i__1].r = 0., h__[i__1].i = 0.;
    }

/*     ==== NBMPS = number of 2-shift bulges in the chain ==== */

    nbmps = ns / 2;

/*     ==== KDU = width of slab ==== */

    kdu = nbmps * 6 - 3;

/*     ==== Create and chase chains of NBMPS bulges ==== */

    i__1 = *kbot - 2;
    i__2 = nbmps * 3 - 2;
    for (incol = (1 - nbmps) * 3 + *ktop - 1; i__2 < 0 ? incol >= i__1 : 
	    incol <= i__1; incol += i__2) {
	ndcol = incol + kdu;
	if (accum) {
	    klaset_("ALL", &kdu, &kdu, &c_b1, &c_b2, &u[u_offset], ldu);
	}

/*        ==== Near-the-diagonal bulge chase.  The following loop */
/*        .    performs the near-the-diagonal part of a small bulge */
/*        .    multi-shift QR sweep.  Each 6*NBMPS-2 column diagonal */
/*        .    chunk extends from column INCOL to column NDCOL */
/*        .    (including both column INCOL and column NDCOL). The */
/*        .    following loop chases a 3*NBMPS column long chain of */
/*        .    NBMPS bulges 3*NBMPS-2 columns to the right.  (INCOL */
/*        .    may be less than KTOP and and NDCOL may be greater than */
/*        .    KBOT indicating phantom columns from which to chase */
/*        .    bulges before they are actually introduced or to which */
/*        .    to chase bulges beyond column KBOT.)  ==== */

/* Computing MIN */
	i__4 = incol + nbmps * 3 - 3, i__5 = *kbot - 2;
	i__3 = f2cmin(i__4,i__5);
	for (krcol = incol; krcol <= i__3; ++krcol) {

/*           ==== Bulges number MTOP to MBOT are active double implicit */
/*           .    shift bulges.  There may or may not also be small */
/*           .    2-by-2 bulge, if there is room.  The inactive bulges */
/*           .    (if any) must wait until the active bulges have moved */
/*           .    down the diagonal to make room.  The phantom matrix */
/*           .    paradigm described above helps keep track.  ==== */

/* Computing MAX */
	    i__4 = 1, i__5 = (*ktop - 1 - krcol + 2) / 3 + 1;
	    mtop = f2cmax(i__4,i__5);
/* Computing MIN */
	    i__4 = nbmps, i__5 = (*kbot - krcol) / 3;
	    mbot = f2cmin(i__4,i__5);
	    m22 = mbot + 1;
	    bmp22 = mbot < nbmps && krcol + (m22 - 1) * 3 == *kbot - 2;

/*           ==== Generate reflections to chase the chain right */
/*           .    one column.  (The minimum value of K is KTOP-1.) ==== */

	    i__4 = mbot;
	    for (m = mtop; m <= i__4; ++m) {
		k = krcol + (m - 1) * 3;
		if (k == *ktop - 1) {
		    klaqr1_(&c__3, &h__[*ktop + *ktop * h_dim1], ldh, &s[(m <<
			     1) - 1], &s[m * 2], &v[m * v_dim1 + 1]);
		    i__5 = m * v_dim1 + 1;
		    alpha.r = v[i__5].r, alpha.i = v[i__5].i;
		    klarfg_(&c__3, &alpha, &v[m * v_dim1 + 2], &c__1, &v[m * 
			    v_dim1 + 1]);
		} else {
		    i__5 = k + 1 + k * h_dim1;
		    beta.r = h__[i__5].r, beta.i = h__[i__5].i;
		    i__5 = m * v_dim1 + 2;
		    i__6 = k + 2 + k * h_dim1;
		    v[i__5].r = h__[i__6].r, v[i__5].i = h__[i__6].i;
		    i__5 = m * v_dim1 + 3;
		    i__6 = k + 3 + k * h_dim1;
		    v[i__5].r = h__[i__6].r, v[i__5].i = h__[i__6].i;
		    klarfg_(&c__3, &beta, &v[m * v_dim1 + 2], &c__1, &v[m * 
			    v_dim1 + 1]);

/*                 ==== A Bulge may collapse because of vigilant */
/*                 .    deflation or destructive underflow.  In the */
/*                 .    underflow case, try the two-small-subdiagonals */
/*                 .    trick to try to reinflate the bulge.  ==== */

		    i__5 = k + 3 + k * h_dim1;
		    i__6 = k + 3 + (k + 1) * h_dim1;
		    i__7 = k + 3 + (k + 2) * h_dim1;
		    if (h__[i__5].r != 0. || h__[i__5].i != 0. || (h__[i__6]
			    .r != 0. || h__[i__6].i != 0.) || h__[i__7].r == 
			    0. && h__[i__7].i == 0.) {

/*                    ==== Typical case: not collapsed (yet). ==== */

			i__5 = k + 1 + k * h_dim1;
			h__[i__5].r = beta.r, h__[i__5].i = beta.i;
			i__5 = k + 2 + k * h_dim1;
			h__[i__5].r = 0., h__[i__5].i = 0.;
			i__5 = k + 3 + k * h_dim1;
			h__[i__5].r = 0., h__[i__5].i = 0.;
		    } else {

/*                    ==== Atypical case: collapsed.  Attempt to */
/*                    .    reintroduce ignoring H(K+1,K) and H(K+2,K). */
/*                    .    If the fill resulting from the new */
/*                    .    reflector is too large, then abandon it. */
/*                    .    Otherwise, use the new one. ==== */

			klaqr1_(&c__3, &h__[k + 1 + (k + 1) * h_dim1], ldh, &
				s[(m << 1) - 1], &s[m * 2], vt);
			alpha.r = vt[0].r, alpha.i = vt[0].i;
			klarfg_(&c__3, &alpha, &vt[1], &c__1, vt);
			d_cnjg(&z__2, vt);
			i__5 = k + 1 + k * h_dim1;
			d_cnjg(&z__5, &vt[1]);
			i__6 = k + 2 + k * h_dim1;
			z__4.r = z__5.r * h__[i__6].r - z__5.i * h__[i__6].i, 
				z__4.i = z__5.r * h__[i__6].i + z__5.i * h__[
				i__6].r;
			z__3.r = h__[i__5].r + z__4.r, z__3.i = h__[i__5].i + 
				z__4.i;
			z__1.r = z__2.r * z__3.r - z__2.i * z__3.i, z__1.i = 
				z__2.r * z__3.i + z__2.i * z__3.r;
			refsum.r = z__1.r, refsum.i = z__1.i;

			i__5 = k + 2 + k * h_dim1;
			z__3.r = refsum.r * vt[1].r - refsum.i * vt[1].i, 
				z__3.i = refsum.r * vt[1].i + refsum.i * vt[1]
				.r;
			z__2.r = h__[i__5].r - z__3.r, z__2.i = h__[i__5].i - 
				z__3.i;
			z__1.r = z__2.r, z__1.i = z__2.i;
			z__5.r = refsum.r * vt[2].r - refsum.i * vt[2].i, 
				z__5.i = refsum.r * vt[2].i + refsum.i * vt[2]
				.r;
			z__4.r = z__5.r, z__4.i = z__5.i;
			i__6 = k + k * h_dim1;
			i__7 = k + 1 + (k + 1) * h_dim1;
			i__8 = k + 2 + (k + 2) * h_dim1;
			if ((d__1 = z__1.r, abs(d__1)) + (d__2 = d_imag(&z__1)
				, abs(d__2)) + ((d__3 = z__4.r, abs(d__3)) + (
				d__4 = d_imag(&z__4), abs(d__4))) > ulp * ((
				d__5 = h__[i__6].r, abs(d__5)) + (d__6 = 
				d_imag(&h__[k + k * h_dim1]), abs(d__6)) + ((
				d__7 = h__[i__7].r, abs(d__7)) + (d__8 = 
				d_imag(&h__[k + 1 + (k + 1) * h_dim1]), abs(
				d__8))) + ((d__9 = h__[i__8].r, abs(d__9)) + (
				d__10 = d_imag(&h__[k + 2 + (k + 2) * h_dim1])
				, abs(d__10))))) {

/*                       ==== Starting a new bulge here would */
/*                       .    create non-negligible fill.  Use */
/*                       .    the old one with trepidation. ==== */

			    i__5 = k + 1 + k * h_dim1;
			    h__[i__5].r = beta.r, h__[i__5].i = beta.i;
			    i__5 = k + 2 + k * h_dim1;
			    h__[i__5].r = 0., h__[i__5].i = 0.;
			    i__5 = k + 3 + k * h_dim1;
			    h__[i__5].r = 0., h__[i__5].i = 0.;
			} else {

/*                       ==== Stating a new bulge here would */
/*                       .    create only negligible fill. */
/*                       .    Replace the old reflector with */
/*                       .    the new one. ==== */

			    i__5 = k + 1 + k * h_dim1;
			    i__6 = k + 1 + k * h_dim1;
			    z__1.r = h__[i__6].r - refsum.r, z__1.i = h__[
				    i__6].i - refsum.i;
			    h__[i__5].r = z__1.r, h__[i__5].i = z__1.i;
			    i__5 = k + 2 + k * h_dim1;
			    h__[i__5].r = 0., h__[i__5].i = 0.;
			    i__5 = k + 3 + k * h_dim1;
			    h__[i__5].r = 0., h__[i__5].i = 0.;
			    i__5 = m * v_dim1 + 1;
			    v[i__5].r = vt[0].r, v[i__5].i = vt[0].i;
			    i__5 = m * v_dim1 + 2;
			    v[i__5].r = vt[1].r, v[i__5].i = vt[1].i;
			    i__5 = m * v_dim1 + 3;
			    v[i__5].r = vt[2].r, v[i__5].i = vt[2].i;
			}
		    }
		}
/* L10: */
	    }

/*           ==== Generate a 2-by-2 reflection, if needed. ==== */

	    k = krcol + (m22 - 1) * 3;
	    if (bmp22) {
		if (k == *ktop - 1) {
		    klaqr1_(&c__2, &h__[k + 1 + (k + 1) * h_dim1], ldh, &s[(
			    m22 << 1) - 1], &s[m22 * 2], &v[m22 * v_dim1 + 1])
			    ;
		    i__4 = m22 * v_dim1 + 1;
		    beta.r = v[i__4].r, beta.i = v[i__4].i;
		    klarfg_(&c__2, &beta, &v[m22 * v_dim1 + 2], &c__1, &v[m22 
			    * v_dim1 + 1]);
		} else {
		    i__4 = k + 1 + k * h_dim1;
		    beta.r = h__[i__4].r, beta.i = h__[i__4].i;
		    i__4 = m22 * v_dim1 + 2;
		    i__5 = k + 2 + k * h_dim1;
		    v[i__4].r = h__[i__5].r, v[i__4].i = h__[i__5].i;
		    klarfg_(&c__2, &beta, &v[m22 * v_dim1 + 2], &c__1, &v[m22 
			    * v_dim1 + 1]);
		    i__4 = k + 1 + k * h_dim1;
		    h__[i__4].r = beta.r, h__[i__4].i = beta.i;
		    i__4 = k + 2 + k * h_dim1;
		    h__[i__4].r = 0., h__[i__4].i = 0.;
		}
	    }

/*           ==== Multiply H by reflections from the left ==== */

	    if (accum) {
		jbot = f2cmin(ndcol,*kbot);
	    } else if (*wantt) {
		jbot = *n;
	    } else {
		jbot = *kbot;
	    }
	    i__4 = jbot;
	    for (j = f2cmax(*ktop,krcol); j <= i__4; ++j) {
/* Computing MIN */
		i__5 = mbot, i__6 = (j - krcol + 2) / 3;
		mend = f2cmin(i__5,i__6);
		i__5 = mend;
		for (m = mtop; m <= i__5; ++m) {
		    k = krcol + (m - 1) * 3;
		    d_cnjg(&z__2, &v[m * v_dim1 + 1]);
		    i__6 = k + 1 + j * h_dim1;
		    d_cnjg(&z__6, &v[m * v_dim1 + 2]);
		    i__7 = k + 2 + j * h_dim1;
		    z__5.r = z__6.r * h__[i__7].r - z__6.i * h__[i__7].i, 
			    z__5.i = z__6.r * h__[i__7].i + z__6.i * h__[i__7]
			    .r;
		    z__4.r = h__[i__6].r + z__5.r, z__4.i = h__[i__6].i + 
			    z__5.i;
		    d_cnjg(&z__8, &v[m * v_dim1 + 3]);
		    i__8 = k + 3 + j * h_dim1;
		    z__7.r = z__8.r * h__[i__8].r - z__8.i * h__[i__8].i, 
			    z__7.i = z__8.r * h__[i__8].i + z__8.i * h__[i__8]
			    .r;
		    z__3.r = z__4.r + z__7.r, z__3.i = z__4.i + z__7.i;
		    z__1.r = z__2.r * z__3.r - z__2.i * z__3.i, z__1.i = 
			    z__2.r * z__3.i + z__2.i * z__3.r;
		    refsum.r = z__1.r, refsum.i = z__1.i;
		    i__6 = k + 1 + j * h_dim1;
		    i__7 = k + 1 + j * h_dim1;
		    z__1.r = h__[i__7].r - refsum.r, z__1.i = h__[i__7].i - 
			    refsum.i;
		    h__[i__6].r = z__1.r, h__[i__6].i = z__1.i;
		    i__6 = k + 2 + j * h_dim1;
		    i__7 = k + 2 + j * h_dim1;
		    i__8 = m * v_dim1 + 2;
		    z__2.r = refsum.r * v[i__8].r - refsum.i * v[i__8].i, 
			    z__2.i = refsum.r * v[i__8].i + refsum.i * v[i__8]
			    .r;
		    z__1.r = h__[i__7].r - z__2.r, z__1.i = h__[i__7].i - 
			    z__2.i;
		    h__[i__6].r = z__1.r, h__[i__6].i = z__1.i;
		    i__6 = k + 3 + j * h_dim1;
		    i__7 = k + 3 + j * h_dim1;
		    i__8 = m * v_dim1 + 3;
		    z__2.r = refsum.r * v[i__8].r - refsum.i * v[i__8].i, 
			    z__2.i = refsum.r * v[i__8].i + refsum.i * v[i__8]
			    .r;
		    z__1.r = h__[i__7].r - z__2.r, z__1.i = h__[i__7].i - 
			    z__2.i;
		    h__[i__6].r = z__1.r, h__[i__6].i = z__1.i;
/* L20: */
		}
/* L30: */
	    }
	    if (bmp22) {
		k = krcol + (m22 - 1) * 3;
/* Computing MAX */
		i__4 = k + 1;
		i__5 = jbot;
		for (j = f2cmax(i__4,*ktop); j <= i__5; ++j) {
		    d_cnjg(&z__2, &v[m22 * v_dim1 + 1]);
		    i__4 = k + 1 + j * h_dim1;
		    d_cnjg(&z__5, &v[m22 * v_dim1 + 2]);
		    i__6 = k + 2 + j * h_dim1;
		    z__4.r = z__5.r * h__[i__6].r - z__5.i * h__[i__6].i, 
			    z__4.i = z__5.r * h__[i__6].i + z__5.i * h__[i__6]
			    .r;
		    z__3.r = h__[i__4].r + z__4.r, z__3.i = h__[i__4].i + 
			    z__4.i;
		    z__1.r = z__2.r * z__3.r - z__2.i * z__3.i, z__1.i = 
			    z__2.r * z__3.i + z__2.i * z__3.r;
		    refsum.r = z__1.r, refsum.i = z__1.i;
		    i__4 = k + 1 + j * h_dim1;
		    i__6 = k + 1 + j * h_dim1;
		    z__1.r = h__[i__6].r - refsum.r, z__1.i = h__[i__6].i - 
			    refsum.i;
		    h__[i__4].r = z__1.r, h__[i__4].i = z__1.i;
		    i__4 = k + 2 + j * h_dim1;
		    i__6 = k + 2 + j * h_dim1;
		    i__7 = m22 * v_dim1 + 2;
		    z__2.r = refsum.r * v[i__7].r - refsum.i * v[i__7].i, 
			    z__2.i = refsum.r * v[i__7].i + refsum.i * v[i__7]
			    .r;
		    z__1.r = h__[i__6].r - z__2.r, z__1.i = h__[i__6].i - 
			    z__2.i;
		    h__[i__4].r = z__1.r, h__[i__4].i = z__1.i;
/* L40: */
		}
	    }

/*           ==== Multiply H by reflections from the right. */
/*           .    Delay filling in the last row until the */
/*           .    vigilant deflation check is complete. ==== */

	    if (accum) {
		jtop = f2cmax(*ktop,incol);
	    } else if (*wantt) {
		jtop = 1;
	    } else {
		jtop = *ktop;
	    }
	    i__5 = mbot;
	    for (m = mtop; m <= i__5; ++m) {
		i__4 = m * v_dim1 + 1;
		if (v[i__4].r != 0. || v[i__4].i != 0.) {
		    k = krcol + (m - 1) * 3;
/* Computing MIN */
		    i__6 = *kbot, i__7 = k + 3;
		    i__4 = f2cmin(i__6,i__7);
		    for (j = jtop; j <= i__4; ++j) {
			i__6 = m * v_dim1 + 1;
			i__7 = j + (k + 1) * h_dim1;
			i__8 = m * v_dim1 + 2;
			i__9 = j + (k + 2) * h_dim1;
			z__4.r = v[i__8].r * h__[i__9].r - v[i__8].i * h__[
				i__9].i, z__4.i = v[i__8].r * h__[i__9].i + v[
				i__8].i * h__[i__9].r;
			z__3.r = h__[i__7].r + z__4.r, z__3.i = h__[i__7].i + 
				z__4.i;
			i__10 = m * v_dim1 + 3;
			i__11 = j + (k + 3) * h_dim1;
			z__5.r = v[i__10].r * h__[i__11].r - v[i__10].i * h__[
				i__11].i, z__5.i = v[i__10].r * h__[i__11].i 
				+ v[i__10].i * h__[i__11].r;
			z__2.r = z__3.r + z__5.r, z__2.i = z__3.i + z__5.i;
			z__1.r = v[i__6].r * z__2.r - v[i__6].i * z__2.i, 
				z__1.i = v[i__6].r * z__2.i + v[i__6].i * 
				z__2.r;
			refsum.r = z__1.r, refsum.i = z__1.i;
			i__6 = j + (k + 1) * h_dim1;
			i__7 = j + (k + 1) * h_dim1;
			z__1.r = h__[i__7].r - refsum.r, z__1.i = h__[i__7].i 
				- refsum.i;
			h__[i__6].r = z__1.r, h__[i__6].i = z__1.i;
			i__6 = j + (k + 2) * h_dim1;
			i__7 = j + (k + 2) * h_dim1;
			d_cnjg(&z__3, &v[m * v_dim1 + 2]);
			z__2.r = refsum.r * z__3.r - refsum.i * z__3.i, 
				z__2.i = refsum.r * z__3.i + refsum.i * 
				z__3.r;
			z__1.r = h__[i__7].r - z__2.r, z__1.i = h__[i__7].i - 
				z__2.i;
			h__[i__6].r = z__1.r, h__[i__6].i = z__1.i;
			i__6 = j + (k + 3) * h_dim1;
			i__7 = j + (k + 3) * h_dim1;
			d_cnjg(&z__3, &v[m * v_dim1 + 3]);
			z__2.r = refsum.r * z__3.r - refsum.i * z__3.i, 
				z__2.i = refsum.r * z__3.i + refsum.i * 
				z__3.r;
			z__1.r = h__[i__7].r - z__2.r, z__1.i = h__[i__7].i - 
				z__2.i;
			h__[i__6].r = z__1.r, h__[i__6].i = z__1.i;
/* L50: */
		    }

		    if (accum) {

/*                    ==== Accumulate U. (If necessary, update Z later */
/*                    .    with with an efficient matrix-matrix */
/*                    .    multiply.) ==== */

			kms = k - incol;
/* Computing MAX */
			i__4 = 1, i__6 = *ktop - incol;
			i__7 = kdu;
			for (j = f2cmax(i__4,i__6); j <= i__7; ++j) {
			    i__4 = m * v_dim1 + 1;
			    i__6 = j + (kms + 1) * u_dim1;
			    i__8 = m * v_dim1 + 2;
			    i__9 = j + (kms + 2) * u_dim1;
			    z__4.r = v[i__8].r * u[i__9].r - v[i__8].i * u[
				    i__9].i, z__4.i = v[i__8].r * u[i__9].i + 
				    v[i__8].i * u[i__9].r;
			    z__3.r = u[i__6].r + z__4.r, z__3.i = u[i__6].i + 
				    z__4.i;
			    i__10 = m * v_dim1 + 3;
			    i__11 = j + (kms + 3) * u_dim1;
			    z__5.r = v[i__10].r * u[i__11].r - v[i__10].i * u[
				    i__11].i, z__5.i = v[i__10].r * u[i__11]
				    .i + v[i__10].i * u[i__11].r;
			    z__2.r = z__3.r + z__5.r, z__2.i = z__3.i + 
				    z__5.i;
			    z__1.r = v[i__4].r * z__2.r - v[i__4].i * z__2.i, 
				    z__1.i = v[i__4].r * z__2.i + v[i__4].i * 
				    z__2.r;
			    refsum.r = z__1.r, refsum.i = z__1.i;
			    i__4 = j + (kms + 1) * u_dim1;
			    i__6 = j + (kms + 1) * u_dim1;
			    z__1.r = u[i__6].r - refsum.r, z__1.i = u[i__6].i 
				    - refsum.i;
			    u[i__4].r = z__1.r, u[i__4].i = z__1.i;
			    i__4 = j + (kms + 2) * u_dim1;
			    i__6 = j + (kms + 2) * u_dim1;
			    d_cnjg(&z__3, &v[m * v_dim1 + 2]);
			    z__2.r = refsum.r * z__3.r - refsum.i * z__3.i, 
				    z__2.i = refsum.r * z__3.i + refsum.i * 
				    z__3.r;
			    z__1.r = u[i__6].r - z__2.r, z__1.i = u[i__6].i - 
				    z__2.i;
			    u[i__4].r = z__1.r, u[i__4].i = z__1.i;
			    i__4 = j + (kms + 3) * u_dim1;
			    i__6 = j + (kms + 3) * u_dim1;
			    d_cnjg(&z__3, &v[m * v_dim1 + 3]);
			    z__2.r = refsum.r * z__3.r - refsum.i * z__3.i, 
				    z__2.i = refsum.r * z__3.i + refsum.i * 
				    z__3.r;
			    z__1.r = u[i__6].r - z__2.r, z__1.i = u[i__6].i - 
				    z__2.i;
			    u[i__4].r = z__1.r, u[i__4].i = z__1.i;
/* L60: */
			}
		    } else if (*wantz) {

/*                    ==== U is not accumulated, so update Z */
/*                    .    now by multiplying by reflections */
/*                    .    from the right. ==== */

			i__7 = *ihiz;
			for (j = *iloz; j <= i__7; ++j) {
			    i__4 = m * v_dim1 + 1;
			    i__6 = j + (k + 1) * z_dim1;
			    i__8 = m * v_dim1 + 2;
			    i__9 = j + (k + 2) * z_dim1;
			    z__4.r = v[i__8].r * z__[i__9].r - v[i__8].i * 
				    z__[i__9].i, z__4.i = v[i__8].r * z__[
				    i__9].i + v[i__8].i * z__[i__9].r;
			    z__3.r = z__[i__6].r + z__4.r, z__3.i = z__[i__6]
				    .i + z__4.i;
			    i__10 = m * v_dim1 + 3;
			    i__11 = j + (k + 3) * z_dim1;
			    z__5.r = v[i__10].r * z__[i__11].r - v[i__10].i * 
				    z__[i__11].i, z__5.i = v[i__10].r * z__[
				    i__11].i + v[i__10].i * z__[i__11].r;
			    z__2.r = z__3.r + z__5.r, z__2.i = z__3.i + 
				    z__5.i;
			    z__1.r = v[i__4].r * z__2.r - v[i__4].i * z__2.i, 
				    z__1.i = v[i__4].r * z__2.i + v[i__4].i * 
				    z__2.r;
			    refsum.r = z__1.r, refsum.i = z__1.i;
			    i__4 = j + (k + 1) * z_dim1;
			    i__6 = j + (k + 1) * z_dim1;
			    z__1.r = z__[i__6].r - refsum.r, z__1.i = z__[
				    i__6].i - refsum.i;
			    z__[i__4].r = z__1.r, z__[i__4].i = z__1.i;
			    i__4 = j + (k + 2) * z_dim1;
			    i__6 = j + (k + 2) * z_dim1;
			    d_cnjg(&z__3, &v[m * v_dim1 + 2]);
			    z__2.r = refsum.r * z__3.r - refsum.i * z__3.i, 
				    z__2.i = refsum.r * z__3.i + refsum.i * 
				    z__3.r;
			    z__1.r = z__[i__6].r - z__2.r, z__1.i = z__[i__6]
				    .i - z__2.i;
			    z__[i__4].r = z__1.r, z__[i__4].i = z__1.i;
			    i__4 = j + (k + 3) * z_dim1;
			    i__6 = j + (k + 3) * z_dim1;
			    d_cnjg(&z__3, &v[m * v_dim1 + 3]);
			    z__2.r = refsum.r * z__3.r - refsum.i * z__3.i, 
				    z__2.i = refsum.r * z__3.i + refsum.i * 
				    z__3.r;
			    z__1.r = z__[i__6].r - z__2.r, z__1.i = z__[i__6]
				    .i - z__2.i;
			    z__[i__4].r = z__1.r, z__[i__4].i = z__1.i;
/* L70: */
			}
		    }
		}
/* L80: */
	    }

/*           ==== Special case: 2-by-2 reflection (if needed) ==== */

	    k = krcol + (m22 - 1) * 3;
	    if (bmp22) {
		i__5 = m22 * v_dim1 + 1;
		if (v[i__5].r != 0. || v[i__5].i != 0.) {
/* Computing MIN */
		    i__7 = *kbot, i__4 = k + 3;
		    i__5 = f2cmin(i__7,i__4);
		    for (j = jtop; j <= i__5; ++j) {
			i__7 = m22 * v_dim1 + 1;
			i__4 = j + (k + 1) * h_dim1;
			i__6 = m22 * v_dim1 + 2;
			i__8 = j + (k + 2) * h_dim1;
			z__3.r = v[i__6].r * h__[i__8].r - v[i__6].i * h__[
				i__8].i, z__3.i = v[i__6].r * h__[i__8].i + v[
				i__6].i * h__[i__8].r;
			z__2.r = h__[i__4].r + z__3.r, z__2.i = h__[i__4].i + 
				z__3.i;
			z__1.r = v[i__7].r * z__2.r - v[i__7].i * z__2.i, 
				z__1.i = v[i__7].r * z__2.i + v[i__7].i * 
				z__2.r;
			refsum.r = z__1.r, refsum.i = z__1.i;
			i__7 = j + (k + 1) * h_dim1;
			i__4 = j + (k + 1) * h_dim1;
			z__1.r = h__[i__4].r - refsum.r, z__1.i = h__[i__4].i 
				- refsum.i;
			h__[i__7].r = z__1.r, h__[i__7].i = z__1.i;
			i__7 = j + (k + 2) * h_dim1;
			i__4 = j + (k + 2) * h_dim1;
			d_cnjg(&z__3, &v[m22 * v_dim1 + 2]);
			z__2.r = refsum.r * z__3.r - refsum.i * z__3.i, 
				z__2.i = refsum.r * z__3.i + refsum.i * 
				z__3.r;
			z__1.r = h__[i__4].r - z__2.r, z__1.i = h__[i__4].i - 
				z__2.i;
			h__[i__7].r = z__1.r, h__[i__7].i = z__1.i;
/* L90: */
		    }

		    if (accum) {
			kms = k - incol;
/* Computing MAX */
			i__5 = 1, i__7 = *ktop - incol;
			i__4 = kdu;
			for (j = f2cmax(i__5,i__7); j <= i__4; ++j) {
			    i__5 = m22 * v_dim1 + 1;
			    i__7 = j + (kms + 1) * u_dim1;
			    i__6 = m22 * v_dim1 + 2;
			    i__8 = j + (kms + 2) * u_dim1;
			    z__3.r = v[i__6].r * u[i__8].r - v[i__6].i * u[
				    i__8].i, z__3.i = v[i__6].r * u[i__8].i + 
				    v[i__6].i * u[i__8].r;
			    z__2.r = u[i__7].r + z__3.r, z__2.i = u[i__7].i + 
				    z__3.i;
			    z__1.r = v[i__5].r * z__2.r - v[i__5].i * z__2.i, 
				    z__1.i = v[i__5].r * z__2.i + v[i__5].i * 
				    z__2.r;
			    refsum.r = z__1.r, refsum.i = z__1.i;
			    i__5 = j + (kms + 1) * u_dim1;
			    i__7 = j + (kms + 1) * u_dim1;
			    z__1.r = u[i__7].r - refsum.r, z__1.i = u[i__7].i 
				    - refsum.i;
			    u[i__5].r = z__1.r, u[i__5].i = z__1.i;
			    i__5 = j + (kms + 2) * u_dim1;
			    i__7 = j + (kms + 2) * u_dim1;
			    d_cnjg(&z__3, &v[m22 * v_dim1 + 2]);
			    z__2.r = refsum.r * z__3.r - refsum.i * z__3.i, 
				    z__2.i = refsum.r * z__3.i + refsum.i * 
				    z__3.r;
			    z__1.r = u[i__7].r - z__2.r, z__1.i = u[i__7].i - 
				    z__2.i;
			    u[i__5].r = z__1.r, u[i__5].i = z__1.i;
/* L100: */
			}
		    } else if (*wantz) {
			i__4 = *ihiz;
			for (j = *iloz; j <= i__4; ++j) {
			    i__5 = m22 * v_dim1 + 1;
			    i__7 = j + (k + 1) * z_dim1;
			    i__6 = m22 * v_dim1 + 2;
			    i__8 = j + (k + 2) * z_dim1;
			    z__3.r = v[i__6].r * z__[i__8].r - v[i__6].i * 
				    z__[i__8].i, z__3.i = v[i__6].r * z__[
				    i__8].i + v[i__6].i * z__[i__8].r;
			    z__2.r = z__[i__7].r + z__3.r, z__2.i = z__[i__7]
				    .i + z__3.i;
			    z__1.r = v[i__5].r * z__2.r - v[i__5].i * z__2.i, 
				    z__1.i = v[i__5].r * z__2.i + v[i__5].i * 
				    z__2.r;
			    refsum.r = z__1.r, refsum.i = z__1.i;
			    i__5 = j + (k + 1) * z_dim1;
			    i__7 = j + (k + 1) * z_dim1;
			    z__1.r = z__[i__7].r - refsum.r, z__1.i = z__[
				    i__7].i - refsum.i;
			    z__[i__5].r = z__1.r, z__[i__5].i = z__1.i;
			    i__5 = j + (k + 2) * z_dim1;
			    i__7 = j + (k + 2) * z_dim1;
			    d_cnjg(&z__3, &v[m22 * v_dim1 + 2]);
			    z__2.r = refsum.r * z__3.r - refsum.i * z__3.i, 
				    z__2.i = refsum.r * z__3.i + refsum.i * 
				    z__3.r;
			    z__1.r = z__[i__7].r - z__2.r, z__1.i = z__[i__7]
				    .i - z__2.i;
			    z__[i__5].r = z__1.r, z__[i__5].i = z__1.i;
/* L110: */
			}
		    }
		}
	    }

/*           ==== Vigilant deflation check ==== */

	    mstart = mtop;
	    if (krcol + (mstart - 1) * 3 < *ktop) {
		++mstart;
	    }
	    mend = mbot;
	    if (bmp22) {
		++mend;
	    }
	    if (krcol == *kbot - 2) {
		++mend;
	    }
	    i__4 = mend;
	    for (m = mstart; m <= i__4; ++m) {
/* Computing MIN */
		i__5 = *kbot - 1, i__7 = krcol + (m - 1) * 3;
		k = f2cmin(i__5,i__7);

/*              ==== The following convergence test requires that */
/*              .    the tradition small-compared-to-nearby-diagonals */
/*              .    criterion and the Ahues & Tisseur (LAWN 122, 1997) */
/*              .    criteria both be satisfied.  The latter improves */
/*              .    accuracy in some examples. Falling back on an */
/*              .    alternate convergence criterion when TST1 or TST2 */
/*              .    is zero (as done here) is traditional but probably */
/*              .    unnecessary. ==== */

		i__5 = k + 1 + k * h_dim1;
		if (h__[i__5].r != 0. || h__[i__5].i != 0.) {
		    i__5 = k + k * h_dim1;
		    i__7 = k + 1 + (k + 1) * h_dim1;
		    tst1 = (d__1 = h__[i__5].r, abs(d__1)) + (d__2 = d_imag(&
			    h__[k + k * h_dim1]), abs(d__2)) + ((d__3 = h__[
			    i__7].r, abs(d__3)) + (d__4 = d_imag(&h__[k + 1 + 
			    (k + 1) * h_dim1]), abs(d__4)));
		    if (tst1 == 0.) {
			if (k >= *ktop + 1) {
			    i__5 = k + (k - 1) * h_dim1;
			    tst1 += (d__1 = h__[i__5].r, abs(d__1)) + (d__2 = 
				    d_imag(&h__[k + (k - 1) * h_dim1]), abs(
				    d__2));
			}
			if (k >= *ktop + 2) {
			    i__5 = k + (k - 2) * h_dim1;
			    tst1 += (d__1 = h__[i__5].r, abs(d__1)) + (d__2 = 
				    d_imag(&h__[k + (k - 2) * h_dim1]), abs(
				    d__2));
			}
			if (k >= *ktop + 3) {
			    i__5 = k + (k - 3) * h_dim1;
			    tst1 += (d__1 = h__[i__5].r, abs(d__1)) + (d__2 = 
				    d_imag(&h__[k + (k - 3) * h_dim1]), abs(
				    d__2));
			}
			if (k <= *kbot - 2) {
			    i__5 = k + 2 + (k + 1) * h_dim1;
			    tst1 += (d__1 = h__[i__5].r, abs(d__1)) + (d__2 = 
				    d_imag(&h__[k + 2 + (k + 1) * h_dim1]), 
				    abs(d__2));
			}
			if (k <= *kbot - 3) {
			    i__5 = k + 3 + (k + 1) * h_dim1;
			    tst1 += (d__1 = h__[i__5].r, abs(d__1)) + (d__2 = 
				    d_imag(&h__[k + 3 + (k + 1) * h_dim1]), 
				    abs(d__2));
			}
			if (k <= *kbot - 4) {
			    i__5 = k + 4 + (k + 1) * h_dim1;
			    tst1 += (d__1 = h__[i__5].r, abs(d__1)) + (d__2 = 
				    d_imag(&h__[k + 4 + (k + 1) * h_dim1]), 
				    abs(d__2));
			}
		    }
		    i__5 = k + 1 + k * h_dim1;
/* Computing MAX */
		    d__3 = smlnum, d__4 = ulp * tst1;
		    if ((d__1 = h__[i__5].r, abs(d__1)) + (d__2 = d_imag(&h__[
			    k + 1 + k * h_dim1]), abs(d__2)) <= f2cmax(d__3,d__4)
			    ) {
/* Computing MAX */
			i__5 = k + 1 + k * h_dim1;
			i__7 = k + (k + 1) * h_dim1;
			d__5 = (d__1 = h__[i__5].r, abs(d__1)) + (d__2 = 
				d_imag(&h__[k + 1 + k * h_dim1]), abs(d__2)), 
				d__6 = (d__3 = h__[i__7].r, abs(d__3)) + (
				d__4 = d_imag(&h__[k + (k + 1) * h_dim1]), 
				abs(d__4));
			h12 = f2cmax(d__5,d__6);
/* Computing MIN */
			i__5 = k + 1 + k * h_dim1;
			i__7 = k + (k + 1) * h_dim1;
			d__5 = (d__1 = h__[i__5].r, abs(d__1)) + (d__2 = 
				d_imag(&h__[k + 1 + k * h_dim1]), abs(d__2)), 
				d__6 = (d__3 = h__[i__7].r, abs(d__3)) + (
				d__4 = d_imag(&h__[k + (k + 1) * h_dim1]), 
				abs(d__4));
			h21 = f2cmin(d__5,d__6);
			i__5 = k + k * h_dim1;
			i__7 = k + 1 + (k + 1) * h_dim1;
			z__2.r = h__[i__5].r - h__[i__7].r, z__2.i = h__[i__5]
				.i - h__[i__7].i;
			z__1.r = z__2.r, z__1.i = z__2.i;
/* Computing MAX */
			i__6 = k + 1 + (k + 1) * h_dim1;
			d__5 = (d__1 = h__[i__6].r, abs(d__1)) + (d__2 = 
				d_imag(&h__[k + 1 + (k + 1) * h_dim1]), abs(
				d__2)), d__6 = (d__3 = z__1.r, abs(d__3)) + (
				d__4 = d_imag(&z__1), abs(d__4));
			h11 = f2cmax(d__5,d__6);
			i__5 = k + k * h_dim1;
			i__7 = k + 1 + (k + 1) * h_dim1;
			z__2.r = h__[i__5].r - h__[i__7].r, z__2.i = h__[i__5]
				.i - h__[i__7].i;
			z__1.r = z__2.r, z__1.i = z__2.i;
/* Computing MIN */
			i__6 = k + 1 + (k + 1) * h_dim1;
			d__5 = (d__1 = h__[i__6].r, abs(d__1)) + (d__2 = 
				d_imag(&h__[k + 1 + (k + 1) * h_dim1]), abs(
				d__2)), d__6 = (d__3 = z__1.r, abs(d__3)) + (
				d__4 = d_imag(&z__1), abs(d__4));
			h22 = f2cmin(d__5,d__6);
			scl = h11 + h12;
			tst2 = h22 * (h11 / scl);

/* Computing MAX */
			d__1 = smlnum, d__2 = ulp * tst2;
			if (tst2 == 0. || h21 * (h12 / scl) <= f2cmax(d__1,d__2))
				 {
			    i__5 = k + 1 + k * h_dim1;
			    h__[i__5].r = 0., h__[i__5].i = 0.;
			}
		    }
		}
/* L120: */
	    }

/*           ==== Fill in the last row of each bulge. ==== */

/* Computing MIN */
	    i__4 = nbmps, i__5 = (*kbot - krcol - 1) / 3;
	    mend = f2cmin(i__4,i__5);
	    i__4 = mend;
	    for (m = mtop; m <= i__4; ++m) {
		k = krcol + (m - 1) * 3;
		i__5 = m * v_dim1 + 1;
		i__7 = m * v_dim1 + 3;
		z__2.r = v[i__5].r * v[i__7].r - v[i__5].i * v[i__7].i, 
			z__2.i = v[i__5].r * v[i__7].i + v[i__5].i * v[i__7]
			.r;
		i__6 = k + 4 + (k + 3) * h_dim1;
		z__1.r = z__2.r * h__[i__6].r - z__2.i * h__[i__6].i, z__1.i =
			 z__2.r * h__[i__6].i + z__2.i * h__[i__6].r;
		refsum.r = z__1.r, refsum.i = z__1.i;
		i__5 = k + 4 + (k + 1) * h_dim1;
		z__1.r = -refsum.r, z__1.i = -refsum.i;
		h__[i__5].r = z__1.r, h__[i__5].i = z__1.i;
		i__5 = k + 4 + (k + 2) * h_dim1;
		z__2.r = -refsum.r, z__2.i = -refsum.i;
		d_cnjg(&z__3, &v[m * v_dim1 + 2]);
		z__1.r = z__2.r * z__3.r - z__2.i * z__3.i, z__1.i = z__2.r * 
			z__3.i + z__2.i * z__3.r;
		h__[i__5].r = z__1.r, h__[i__5].i = z__1.i;
		i__5 = k + 4 + (k + 3) * h_dim1;
		i__7 = k + 4 + (k + 3) * h_dim1;
		d_cnjg(&z__3, &v[m * v_dim1 + 3]);
		z__2.r = refsum.r * z__3.r - refsum.i * z__3.i, z__2.i = 
			refsum.r * z__3.i + refsum.i * z__3.r;
		z__1.r = h__[i__7].r - z__2.r, z__1.i = h__[i__7].i - z__2.i;
		h__[i__5].r = z__1.r, h__[i__5].i = z__1.i;
/* L130: */
	    }

/*           ==== End of near-the-diagonal bulge chase. ==== */

/* L140: */
	}

/*        ==== Use U (if accumulated) to update far-from-diagonal */
/*        .    entries in H.  If required, use U to update Z as */
/*        .    well. ==== */

	if (accum) {
	    if (*wantt) {
		jtop = 1;
		jbot = *n;
	    } else {
		jtop = *ktop;
		jbot = *kbot;
	    }
	    if (! blk22 || incol < *ktop || ndcol > *kbot || ns <= 2) {

/*              ==== Updates not exploiting the 2-by-2 block */
/*              .    structure of U.  K1 and NU keep track of */
/*              .    the location and size of U in the special */
/*              .    cases of introducing bulges and chasing */
/*              .    bulges off the bottom.  In these special */
/*              .    cases and in case the number of shifts */
/*              .    is NS = 2, there is no 2-by-2 block */
/*              .    structure to exploit.  ==== */

/* Computing MAX */
		i__3 = 1, i__4 = *ktop - incol;
		k1 = f2cmax(i__3,i__4);
/* Computing MAX */
		i__3 = 0, i__4 = ndcol - *kbot;
		nu = kdu - f2cmax(i__3,i__4) - k1 + 1;

/*              ==== Horizontal Multiply ==== */

		i__3 = jbot;
		i__4 = *nh;
		for (jcol = f2cmin(ndcol,*kbot) + 1; i__4 < 0 ? jcol >= i__3 : 
			jcol <= i__3; jcol += i__4) {
/* Computing MIN */
		    i__5 = *nh, i__7 = jbot - jcol + 1;
		    jlen = f2cmin(i__5,i__7);
		    kgemm_("C", "N", &nu, &jlen, &nu, &c_b2, &u[k1 + k1 * 
			    u_dim1], ldu, &h__[incol + k1 + jcol * h_dim1], 
			    ldh, &c_b1, &wh[wh_offset], ldwh);
		    klacpy_("ALL", &nu, &jlen, &wh[wh_offset], ldwh, &h__[
			    incol + k1 + jcol * h_dim1], ldh);
/* L150: */
		}

/*              ==== Vertical multiply ==== */

		i__4 = f2cmax(*ktop,incol) - 1;
		i__3 = *nv;
		for (jrow = jtop; i__3 < 0 ? jrow >= i__4 : jrow <= i__4; 
			jrow += i__3) {
/* Computing MIN */
		    i__5 = *nv, i__7 = f2cmax(*ktop,incol) - jrow;
		    jlen = f2cmin(i__5,i__7);
		    kgemm_("N", "N", &jlen, &nu, &nu, &c_b2, &h__[jrow + (
			    incol + k1) * h_dim1], ldh, &u[k1 + k1 * u_dim1], 
			    ldu, &c_b1, &wv[wv_offset], ldwv);
		    klacpy_("ALL", &jlen, &nu, &wv[wv_offset], ldwv, &h__[
			    jrow + (incol + k1) * h_dim1], ldh);
/* L160: */
		}

/*              ==== Z multiply (also vertical) ==== */

		if (*wantz) {
		    i__3 = *ihiz;
		    i__4 = *nv;
		    for (jrow = *iloz; i__4 < 0 ? jrow >= i__3 : jrow <= i__3;
			     jrow += i__4) {
/* Computing MIN */
			i__5 = *nv, i__7 = *ihiz - jrow + 1;
			jlen = f2cmin(i__5,i__7);
			kgemm_("N", "N", &jlen, &nu, &nu, &c_b2, &z__[jrow + (
				incol + k1) * z_dim1], ldz, &u[k1 + k1 * 
				u_dim1], ldu, &c_b1, &wv[wv_offset], ldwv);
			klacpy_("ALL", &jlen, &nu, &wv[wv_offset], ldwv, &z__[
				jrow + (incol + k1) * z_dim1], ldz)
				;
/* L170: */
		    }
		}
	    } else {

/*              ==== Updates exploiting U's 2-by-2 block structure. */
/*              .    (I2, I4, J2, J4 are the last rows and columns */
/*              .    of the blocks.) ==== */

		i2 = (kdu + 1) / 2;
		i4 = kdu;
		j2 = i4 - i2;
		j4 = kdu;

/*              ==== KZS and KNZ deal with the band of zeros */
/*              .    along the diagonal of one of the triangular */
/*              .    blocks. ==== */

		kzs = j4 - j2 - (ns + 1);
		knz = ns + 1;

/*              ==== Horizontal multiply ==== */

		i__4 = jbot;
		i__3 = *nh;
		for (jcol = f2cmin(ndcol,*kbot) + 1; i__3 < 0 ? jcol >= i__4 : 
			jcol <= i__4; jcol += i__3) {
/* Computing MIN */
		    i__5 = *nh, i__7 = jbot - jcol + 1;
		    jlen = f2cmin(i__5,i__7);

/*                 ==== Copy bottom of H to top+KZS of scratch ==== */
/*                  (The first KZS rows get multiplied by zero.) ==== */

		    klacpy_("ALL", &knz, &jlen, &h__[incol + 1 + j2 + jcol * 
			    h_dim1], ldh, &wh[kzs + 1 + wh_dim1], ldwh);

/*                 ==== Multiply by U21**H ==== */

		    klaset_("ALL", &kzs, &jlen, &c_b1, &c_b1, &wh[wh_offset], 
			    ldwh);
		    ktrmm_("L", "U", "C", "N", &knz, &jlen, &c_b2, &u[j2 + 1 
			    + (kzs + 1) * u_dim1], ldu, &wh[kzs + 1 + wh_dim1]
			    , ldwh);

/*                 ==== Multiply top of H by U11**H ==== */

		    kgemm_("C", "N", &i2, &jlen, &j2, &c_b2, &u[u_offset], 
			    ldu, &h__[incol + 1 + jcol * h_dim1], ldh, &c_b2, 
			    &wh[wh_offset], ldwh);

/*                 ==== Copy top of H to bottom of WH ==== */

		    klacpy_("ALL", &j2, &jlen, &h__[incol + 1 + jcol * h_dim1]
			    , ldh, &wh[i2 + 1 + wh_dim1], ldwh);

/*                 ==== Multiply by U21**H ==== */

		    ktrmm_("L", "L", "C", "N", &j2, &jlen, &c_b2, &u[(i2 + 1) 
			    * u_dim1 + 1], ldu, &wh[i2 + 1 + wh_dim1], ldwh);

/*                 ==== Multiply by U22 ==== */

		    i__5 = i4 - i2;
		    i__7 = j4 - j2;
		    kgemm_("C", "N", &i__5, &jlen, &i__7, &c_b2, &u[j2 + 1 + (
			    i2 + 1) * u_dim1], ldu, &h__[incol + 1 + j2 + 
			    jcol * h_dim1], ldh, &c_b2, &wh[i2 + 1 + wh_dim1],
			     ldwh);

/*                 ==== Copy it back ==== */

		    klacpy_("ALL", &kdu, &jlen, &wh[wh_offset], ldwh, &h__[
			    incol + 1 + jcol * h_dim1], ldh);
/* L180: */
		}

/*              ==== Vertical multiply ==== */

		i__3 = f2cmax(incol,*ktop) - 1;
		i__4 = *nv;
		for (jrow = jtop; i__4 < 0 ? jrow >= i__3 : jrow <= i__3; 
			jrow += i__4) {
/* Computing MIN */
		    i__5 = *nv, i__7 = f2cmax(incol,*ktop) - jrow;
		    jlen = f2cmin(i__5,i__7);

/*                 ==== Copy right of H to scratch (the first KZS */
/*                 .    columns get multiplied by zero) ==== */

		    klacpy_("ALL", &jlen, &knz, &h__[jrow + (incol + 1 + j2) *
			     h_dim1], ldh, &wv[(kzs + 1) * wv_dim1 + 1], ldwv);

/*                 ==== Multiply by U21 ==== */

		    klaset_("ALL", &jlen, &kzs, &c_b1, &c_b1, &wv[wv_offset], 
			    ldwv);
		    ktrmm_("R", "U", "N", "N", &jlen, &knz, &c_b2, &u[j2 + 1 
			    + (kzs + 1) * u_dim1], ldu, &wv[(kzs + 1) * 
			    wv_dim1 + 1], ldwv);

/*                 ==== Multiply by U11 ==== */

		    kgemm_("N", "N", &jlen, &i2, &j2, &c_b2, &h__[jrow + (
			    incol + 1) * h_dim1], ldh, &u[u_offset], ldu, &
			    c_b2, &wv[wv_offset], ldwv);

/*                 ==== Copy left of H to right of scratch ==== */

		    klacpy_("ALL", &jlen, &j2, &h__[jrow + (incol + 1) * 
			    h_dim1], ldh, &wv[(i2 + 1) * wv_dim1 + 1], ldwv);

/*                 ==== Multiply by U21 ==== */

		    i__5 = i4 - i2;
		    ktrmm_("R", "L", "N", "N", &jlen, &i__5, &c_b2, &u[(i2 + 
			    1) * u_dim1 + 1], ldu, &wv[(i2 + 1) * wv_dim1 + 1]
			    , ldwv);

/*                 ==== Multiply by U22 ==== */

		    i__5 = i4 - i2;
		    i__7 = j4 - j2;
		    kgemm_("N", "N", &jlen, &i__5, &i__7, &c_b2, &h__[jrow + (
			    incol + 1 + j2) * h_dim1], ldh, &u[j2 + 1 + (i2 + 
			    1) * u_dim1], ldu, &c_b2, &wv[(i2 + 1) * wv_dim1 
			    + 1], ldwv);

/*                 ==== Copy it back ==== */

		    klacpy_("ALL", &jlen, &kdu, &wv[wv_offset], ldwv, &h__[
			    jrow + (incol + 1) * h_dim1], ldh);
/* L190: */
		}

/*              ==== Multiply Z (also vertical) ==== */

		if (*wantz) {
		    i__4 = *ihiz;
		    i__3 = *nv;
		    for (jrow = *iloz; i__3 < 0 ? jrow >= i__4 : jrow <= i__4;
			     jrow += i__3) {
/* Computing MIN */
			i__5 = *nv, i__7 = *ihiz - jrow + 1;
			jlen = f2cmin(i__5,i__7);

/*                    ==== Copy right of Z to left of scratch (first */
/*                    .     KZS columns get multiplied by zero) ==== */

			klacpy_("ALL", &jlen, &knz, &z__[jrow + (incol + 1 + 
				j2) * z_dim1], ldz, &wv[(kzs + 1) * wv_dim1 + 
				1], ldwv);

/*                    ==== Multiply by U12 ==== */

			klaset_("ALL", &jlen, &kzs, &c_b1, &c_b1, &wv[
				wv_offset], ldwv);
			ktrmm_("R", "U", "N", "N", &jlen, &knz, &c_b2, &u[j2 
				+ 1 + (kzs + 1) * u_dim1], ldu, &wv[(kzs + 1) 
				* wv_dim1 + 1], ldwv);

/*                    ==== Multiply by U11 ==== */

			kgemm_("N", "N", &jlen, &i2, &j2, &c_b2, &z__[jrow + (
				incol + 1) * z_dim1], ldz, &u[u_offset], ldu, 
				&c_b2, &wv[wv_offset], ldwv);

/*                    ==== Copy left of Z to right of scratch ==== */

			klacpy_("ALL", &jlen, &j2, &z__[jrow + (incol + 1) * 
				z_dim1], ldz, &wv[(i2 + 1) * wv_dim1 + 1], 
				ldwv);

/*                    ==== Multiply by U21 ==== */

			i__5 = i4 - i2;
			ktrmm_("R", "L", "N", "N", &jlen, &i__5, &c_b2, &u[(
				i2 + 1) * u_dim1 + 1], ldu, &wv[(i2 + 1) * 
				wv_dim1 + 1], ldwv);

/*                    ==== Multiply by U22 ==== */

			i__5 = i4 - i2;
			i__7 = j4 - j2;
			kgemm_("N", "N", &jlen, &i__5, &i__7, &c_b2, &z__[
				jrow + (incol + 1 + j2) * z_dim1], ldz, &u[j2 
				+ 1 + (i2 + 1) * u_dim1], ldu, &c_b2, &wv[(i2 
				+ 1) * wv_dim1 + 1], ldwv);

/*                    ==== Copy the result back to Z ==== */

			klacpy_("ALL", &jlen, &kdu, &wv[wv_offset], ldwv, &
				z__[jrow + (incol + 1) * z_dim1], ldz);
/* L200: */
		    }
		}
	    }
	}
/* L210: */
    }

/*     ==== End of ZLAQR5 ==== */

    return;
} /* klaqr5_ */

