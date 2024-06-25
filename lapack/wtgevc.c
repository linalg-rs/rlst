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

static quadcomplex c_b1 = {0.,0.};
static quadcomplex c_b2 = {1.,0.};
static integer c__1 = 1;

/* > \brief \b ZTGEVC */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZTGEVC + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/ztgevc.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/ztgevc.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/ztgevc.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZTGEVC( SIDE, HOWMNY, SELECT, N, S, LDS, P, LDP, VL, */
/*                          LDVL, VR, LDVR, MM, M, WORK, RWORK, INFO ) */

/*       CHARACTER          HOWMNY, SIDE */
/*       INTEGER            INFO, LDP, LDS, LDVL, LDVR, M, MM, N */
/*       LOGICAL            SELECT( * ) */
/*       DOUBLE PRECISION   RWORK( * ) */
/*       COMPLEX*16         P( LDP, * ), S( LDS, * ), VL( LDVL, * ), */
/*      $                   VR( LDVR, * ), WORK( * ) */



/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZTGEVC computes some or all of the right and/or left eigenvectors of */
/* > a pair of complex matrices (S,P), where S and P are upper triangular. */
/* > Matrix pairs of this type are produced by the generalized Schur */
/* > factorization of a complex matrix pair (A,B): */
/* > */
/* >    A = Q*S*Z**H,  B = Q*P*Z**H */
/* > */
/* > as computed by ZGGHRD + ZHGEQZ. */
/* > */
/* > The right eigenvector x and the left eigenvector y of (S,P) */
/* > corresponding to an eigenvalue w are defined by: */
/* > */
/* >    S*x = w*P*x,  (y**H)*S = w*(y**H)*P, */
/* > */
/* > where y**H denotes the conjugate tranpose of y. */
/* > The eigenvalues are not input to this routine, but are computed */
/* > directly from the diagonal elements of S and P. */
/* > */
/* > This routine returns the matrices X and/or Y of right and left */
/* > eigenvectors of (S,P), or the products Z*X and/or Q*Y, */
/* > where Z and Q are input matrices. */
/* > If Q and Z are the unitary factors from the generalized Schur */
/* > factorization of a matrix pair (A,B), then Z*X and Q*Y */
/* > are the matrices of right and left eigenvectors of (A,B). */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] SIDE */
/* > \verbatim */
/* >          SIDE is CHARACTER*1 */
/* >          = 'R': compute right eigenvectors only; */
/* >          = 'L': compute left eigenvectors only; */
/* >          = 'B': compute both right and left eigenvectors. */
/* > \endverbatim */
/* > */
/* > \param[in] HOWMNY */
/* > \verbatim */
/* >          HOWMNY is CHARACTER*1 */
/* >          = 'A': compute all right and/or left eigenvectors; */
/* >          = 'B': compute all right and/or left eigenvectors, */
/* >                 backtransformed by the matrices in VR and/or VL; */
/* >          = 'S': compute selected right and/or left eigenvectors, */
/* >                 specified by the logical array SELECT. */
/* > \endverbatim */
/* > */
/* > \param[in] SELECT */
/* > \verbatim */
/* >          SELECT is LOGICAL array, dimension (N) */
/* >          If HOWMNY='S', SELECT specifies the eigenvectors to be */
/* >          computed.  The eigenvector corresponding to the j-th */
/* >          eigenvalue is computed if SELECT(j) = .TRUE.. */
/* >          Not referenced if HOWMNY = 'A' or 'B'. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrices S and P.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] S */
/* > \verbatim */
/* >          S is COMPLEX*16 array, dimension (LDS,N) */
/* >          The upper triangular matrix S from a generalized Schur */
/* >          factorization, as computed by ZHGEQZ. */
/* > \endverbatim */
/* > */
/* > \param[in] LDS */
/* > \verbatim */
/* >          LDS is INTEGER */
/* >          The leading dimension of array S.  LDS >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in] P */
/* > \verbatim */
/* >          P is COMPLEX*16 array, dimension (LDP,N) */
/* >          The upper triangular matrix P from a generalized Schur */
/* >          factorization, as computed by ZHGEQZ.  P must have doublereal */
/* >          diagonal elements. */
/* > \endverbatim */
/* > */
/* > \param[in] LDP */
/* > \verbatim */
/* >          LDP is INTEGER */
/* >          The leading dimension of array P.  LDP >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in,out] VL */
/* > \verbatim */
/* >          VL is COMPLEX*16 array, dimension (LDVL,MM) */
/* >          On entry, if SIDE = 'L' or 'B' and HOWMNY = 'B', VL must */
/* >          contain an N-by-N matrix Q (usually the unitary matrix Q */
/* >          of left Schur vectors returned by ZHGEQZ). */
/* >          On exit, if SIDE = 'L' or 'B', VL contains: */
/* >          if HOWMNY = 'A', the matrix Y of left eigenvectors of (S,P); */
/* >          if HOWMNY = 'B', the matrix Q*Y; */
/* >          if HOWMNY = 'S', the left eigenvectors of (S,P) specified by */
/* >                      SELECT, stored consecutively in the columns of */
/* >                      VL, in the same order as their eigenvalues. */
/* >          Not referenced if SIDE = 'R'. */
/* > \endverbatim */
/* > */
/* > \param[in] LDVL */
/* > \verbatim */
/* >          LDVL is INTEGER */
/* >          The leading dimension of array VL.  LDVL >= 1, and if */
/* >          SIDE = 'L' or 'l' or 'B' or 'b', LDVL >= N. */
/* > \endverbatim */
/* > */
/* > \param[in,out] VR */
/* > \verbatim */
/* >          VR is COMPLEX*16 array, dimension (LDVR,MM) */
/* >          On entry, if SIDE = 'R' or 'B' and HOWMNY = 'B', VR must */
/* >          contain an N-by-N matrix Q (usually the unitary matrix Z */
/* >          of right Schur vectors returned by ZHGEQZ). */
/* >          On exit, if SIDE = 'R' or 'B', VR contains: */
/* >          if HOWMNY = 'A', the matrix X of right eigenvectors of (S,P); */
/* >          if HOWMNY = 'B', the matrix Z*X; */
/* >          if HOWMNY = 'S', the right eigenvectors of (S,P) specified by */
/* >                      SELECT, stored consecutively in the columns of */
/* >                      VR, in the same order as their eigenvalues. */
/* >          Not referenced if SIDE = 'L'. */
/* > \endverbatim */
/* > */
/* > \param[in] LDVR */
/* > \verbatim */
/* >          LDVR is INTEGER */
/* >          The leading dimension of the array VR.  LDVR >= 1, and if */
/* >          SIDE = 'R' or 'B', LDVR >= N. */
/* > \endverbatim */
/* > */
/* > \param[in] MM */
/* > \verbatim */
/* >          MM is INTEGER */
/* >          The number of columns in the arrays VL and/or VR. MM >= M. */
/* > \endverbatim */
/* > */
/* > \param[out] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of columns in the arrays VL and/or VR actually */
/* >          used to store the eigenvectors.  If HOWMNY = 'A' or 'B', M */
/* >          is set to N.  Each selected eigenvector occupies one column. */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is COMPLEX*16 array, dimension (2*N) */
/* > \endverbatim */
/* > */
/* > \param[out] RWORK */
/* > \verbatim */
/* >          RWORK is DOUBLE PRECISION array, dimension (2*N) */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit. */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value. */
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
void  wtgevc_(char *side, char *howmny, logical *select, 
	integer *n, quadcomplex *s, integer *lds, quadcomplex *p, integer 
	*ldp, quadcomplex *vl, integer *ldvl, quadcomplex *vr, integer *
	ldvr, integer *mm, integer *m, quadcomplex *work, quadreal *rwork,
	 integer *info)
{
    /* System generated locals */
    integer p_dim1, p_offset, s_dim1, s_offset, vl_dim1, vl_offset, vr_dim1, 
	    vr_offset, i__1, i__2, i__3, i__4, i__5;
    quadreal d__1, d__2, d__3, d__4, d__5, d__6;
    quadcomplex z__1, z__2, z__3, z__4;

    /* Local variables */
    quadcomplex d__;
    integer i__, j;
    quadcomplex ca, cb;
    integer je, im, jr;
    quadreal big;
    logical lsa, lsb;
    quadreal ulp;
    quadcomplex sum;
    integer ibeg, ieig, iend;
    quadreal dmin__;
    integer isrc;
    quadreal temp;
    quadcomplex suma, sumb;
    quadreal xmax, scale;
    logical ilall;
    integer iside;
    quadreal sbeta;
    extern logical lsame_(char *, char *);
    quadreal small;
    logical compl;
    quadreal anorm, bnorm;
    logical compr;
    extern void  wgemv_(char *, integer *, integer *, 
	    quadcomplex *, quadcomplex *, integer *, quadcomplex *, 
	    integer *, quadcomplex *, quadcomplex *, integer *), 
	    qlabad_(quadreal *, quadreal *);
    logical ilbbad;
    quadreal acoefa, bcoefa, acoeff;
    quadcomplex bcoeff;
    logical ilback;
    quadreal ascale, bscale;
    extern quadreal qlamch_(char *);
    quadcomplex salpha;
    quadreal safmin;
    extern void  xerbla_(char *, integer *);
    quadreal bignum;
    logical ilcomp;
    extern /* Double Complex */ VOID wladiv_(quadcomplex *, quadcomplex *,
	     quadcomplex *);
    integer ihwmny;


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */



/*  ===================================================================== */


/*     Decode and Test the input parameters */

    /* Parameter adjustments */
    --select;
    s_dim1 = *lds;
    s_offset = 1 + s_dim1;
    s -= s_offset;
    p_dim1 = *ldp;
    p_offset = 1 + p_dim1;
    p -= p_offset;
    vl_dim1 = *ldvl;
    vl_offset = 1 + vl_dim1;
    vl -= vl_offset;
    vr_dim1 = *ldvr;
    vr_offset = 1 + vr_dim1;
    vr -= vr_offset;
    --work;
    --rwork;

    /* Function Body */
    if (lsame_(howmny, "A")) {
	ihwmny = 1;
	ilall = TRUE_;
	ilback = FALSE_;
    } else if (lsame_(howmny, "S")) {
	ihwmny = 2;
	ilall = FALSE_;
	ilback = FALSE_;
    } else if (lsame_(howmny, "B")) {
	ihwmny = 3;
	ilall = TRUE_;
	ilback = TRUE_;
    } else {
	ihwmny = -1;
    }

    if (lsame_(side, "R")) {
	iside = 1;
	compl = FALSE_;
	compr = TRUE_;
    } else if (lsame_(side, "L")) {
	iside = 2;
	compl = TRUE_;
	compr = FALSE_;
    } else if (lsame_(side, "B")) {
	iside = 3;
	compl = TRUE_;
	compr = TRUE_;
    } else {
	iside = -1;
    }

    *info = 0;
    if (iside < 0) {
	*info = -1;
    } else if (ihwmny < 0) {
	*info = -2;
    } else if (*n < 0) {
	*info = -4;
    } else if (*lds < f2cmax(1,*n)) {
	*info = -6;
    } else if (*ldp < f2cmax(1,*n)) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZTGEVC", &i__1);
	return;
    }

/*     Count the number of eigenvectors */

    if (! ilall) {
	im = 0;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (select[j]) {
		++im;
	    }
/* L10: */
	}
    } else {
	im = *n;
    }

/*     Check diagonal of B */

    ilbbad = FALSE_;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	if (d_imag(&p[j + j * p_dim1]) != 0.) {
	    ilbbad = TRUE_;
	}
/* L20: */
    }

    if (ilbbad) {
	*info = -7;
    } else if (compl && *ldvl < *n || *ldvl < 1) {
	*info = -10;
    } else if (compr && *ldvr < *n || *ldvr < 1) {
	*info = -12;
    } else if (*mm < im) {
	*info = -13;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZTGEVC", &i__1);
	return;
    }

/*     Quick return if possible */

    *m = im;
    if (*n == 0) {
	return;
    }

/*     Machine Constants */

    safmin = qlamch_("Safe minimum");
    big = 1. / safmin;
    qlabad_(&safmin, &big);
    ulp = qlamch_("Epsilon") * qlamch_("Base");
    small = safmin * *n / ulp;
    big = 1. / small;
    bignum = 1. / (safmin * *n);

/*     Compute the 1-norm of each column of the strictly upper triangular */
/*     part of A and B to check for possible overflow in the triangular */
/*     solver. */

    i__1 = s_dim1 + 1;
    anorm = (d__1 = s[i__1].r, abs(d__1)) + (d__2 = d_imag(&s[s_dim1 + 1]), 
	    abs(d__2));
    i__1 = p_dim1 + 1;
    bnorm = (d__1 = p[i__1].r, abs(d__1)) + (d__2 = d_imag(&p[p_dim1 + 1]), 
	    abs(d__2));
    rwork[1] = 0.;
    rwork[*n + 1] = 0.;
    i__1 = *n;
    for (j = 2; j <= i__1; ++j) {
	rwork[j] = 0.;
	rwork[*n + j] = 0.;
	i__2 = j - 1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    i__3 = i__ + j * s_dim1;
	    rwork[j] += (d__1 = s[i__3].r, abs(d__1)) + (d__2 = d_imag(&s[i__ 
		    + j * s_dim1]), abs(d__2));
	    i__3 = i__ + j * p_dim1;
	    rwork[*n + j] += (d__1 = p[i__3].r, abs(d__1)) + (d__2 = d_imag(&
		    p[i__ + j * p_dim1]), abs(d__2));
/* L30: */
	}
/* Computing MAX */
	i__2 = j + j * s_dim1;
	d__3 = anorm, d__4 = rwork[j] + ((d__1 = s[i__2].r, abs(d__1)) + (
		d__2 = d_imag(&s[j + j * s_dim1]), abs(d__2)));
	anorm = f2cmax(d__3,d__4);
/* Computing MAX */
	i__2 = j + j * p_dim1;
	d__3 = bnorm, d__4 = rwork[*n + j] + ((d__1 = p[i__2].r, abs(d__1)) + 
		(d__2 = d_imag(&p[j + j * p_dim1]), abs(d__2)));
	bnorm = f2cmax(d__3,d__4);
/* L40: */
    }

    ascale = 1. / f2cmax(anorm,safmin);
    bscale = 1. / f2cmax(bnorm,safmin);

/*     Left eigenvectors */

    if (compl) {
	ieig = 0;

/*        Main loop over eigenvalues */

	i__1 = *n;
	for (je = 1; je <= i__1; ++je) {
	    if (ilall) {
		ilcomp = TRUE_;
	    } else {
		ilcomp = select[je];
	    }
	    if (ilcomp) {
		++ieig;

		i__2 = je + je * s_dim1;
		i__3 = je + je * p_dim1;
		if ((d__2 = s[i__2].r, abs(d__2)) + (d__3 = d_imag(&s[je + je 
			* s_dim1]), abs(d__3)) <= safmin && (d__1 = p[i__3].r,
			 abs(d__1)) <= safmin) {

/*                 Singular matrix pencil -- return unit eigenvector */

		    i__2 = *n;
		    for (jr = 1; jr <= i__2; ++jr) {
			i__3 = jr + ieig * vl_dim1;
			vl[i__3].r = 0., vl[i__3].i = 0.;
/* L50: */
		    }
		    i__2 = ieig + ieig * vl_dim1;
		    vl[i__2].r = 1., vl[i__2].i = 0.;
		    goto L140;
		}

/*              Non-singular eigenvalue: */
/*              Compute coefficients  a  and  b  in */
/*                   H */
/*                 y  ( a A - b B ) = 0 */

/* Computing MAX */
		i__2 = je + je * s_dim1;
		i__3 = je + je * p_dim1;
		d__4 = ((d__2 = s[i__2].r, abs(d__2)) + (d__3 = d_imag(&s[je 
			+ je * s_dim1]), abs(d__3))) * ascale, d__5 = (d__1 = 
			p[i__3].r, abs(d__1)) * bscale, d__4 = f2cmax(d__4,d__5);
		temp = 1. / f2cmax(d__4,safmin);
		i__2 = je + je * s_dim1;
		z__2.r = temp * s[i__2].r, z__2.i = temp * s[i__2].i;
		z__1.r = ascale * z__2.r, z__1.i = ascale * z__2.i;
		salpha.r = z__1.r, salpha.i = z__1.i;
		i__2 = je + je * p_dim1;
		sbeta = temp * p[i__2].r * bscale;
		acoeff = sbeta * ascale;
		z__1.r = bscale * salpha.r, z__1.i = bscale * salpha.i;
		bcoeff.r = z__1.r, bcoeff.i = z__1.i;

/*              Scale to avoid underflow */

		lsa = abs(sbeta) >= safmin && abs(acoeff) < small;
		lsb = (d__1 = salpha.r, abs(d__1)) + (d__2 = d_imag(&salpha), 
			abs(d__2)) >= safmin && (d__3 = bcoeff.r, abs(d__3)) 
			+ (d__4 = d_imag(&bcoeff), abs(d__4)) < small;

		scale = 1.;
		if (lsa) {
		    scale = small / abs(sbeta) * f2cmin(anorm,big);
		}
		if (lsb) {
/* Computing MAX */
		    d__3 = scale, d__4 = small / ((d__1 = salpha.r, abs(d__1))
			     + (d__2 = d_imag(&salpha), abs(d__2))) * f2cmin(
			    bnorm,big);
		    scale = f2cmax(d__3,d__4);
		}
		if (lsa || lsb) {
/* Computing MIN */
/* Computing MAX */
		    d__5 = 1., d__6 = abs(acoeff), d__5 = f2cmax(d__5,d__6), 
			    d__6 = (d__1 = bcoeff.r, abs(d__1)) + (d__2 = 
			    d_imag(&bcoeff), abs(d__2));
		    d__3 = scale, d__4 = 1. / (safmin * f2cmax(d__5,d__6));
		    scale = f2cmin(d__3,d__4);
		    if (lsa) {
			acoeff = ascale * (scale * sbeta);
		    } else {
			acoeff = scale * acoeff;
		    }
		    if (lsb) {
			z__2.r = scale * salpha.r, z__2.i = scale * salpha.i;
			z__1.r = bscale * z__2.r, z__1.i = bscale * z__2.i;
			bcoeff.r = z__1.r, bcoeff.i = z__1.i;
		    } else {
			z__1.r = scale * bcoeff.r, z__1.i = scale * bcoeff.i;
			bcoeff.r = z__1.r, bcoeff.i = z__1.i;
		    }
		}

		acoefa = abs(acoeff);
		bcoefa = (d__1 = bcoeff.r, abs(d__1)) + (d__2 = d_imag(&
			bcoeff), abs(d__2));
		xmax = 1.;
		i__2 = *n;
		for (jr = 1; jr <= i__2; ++jr) {
		    i__3 = jr;
		    work[i__3].r = 0., work[i__3].i = 0.;
/* L60: */
		}
		i__2 = je;
		work[i__2].r = 1., work[i__2].i = 0.;
/* Computing MAX */
		d__1 = ulp * acoefa * anorm, d__2 = ulp * bcoefa * bnorm, 
			d__1 = f2cmax(d__1,d__2);
		dmin__ = f2cmax(d__1,safmin);

/*                                              H */
/*              Triangular solve of  (a A - b B)  y = 0 */

/*                                      H */
/*              (rowwise in  (a A - b B) , or columnwise in a A - b B) */

		i__2 = *n;
		for (j = je + 1; j <= i__2; ++j) {

/*                 Compute */
/*                       j-1 */
/*                 SUM = sum  conjg( a*S(k,j) - b*P(k,j) )*x(k) */
/*                       k=je */
/*                 (Scale if necessary) */

		    temp = 1. / xmax;
		    if (acoefa * rwork[j] + bcoefa * rwork[*n + j] > bignum * 
			    temp) {
			i__3 = j - 1;
			for (jr = je; jr <= i__3; ++jr) {
			    i__4 = jr;
			    i__5 = jr;
			    z__1.r = temp * work[i__5].r, z__1.i = temp * 
				    work[i__5].i;
			    work[i__4].r = z__1.r, work[i__4].i = z__1.i;
/* L70: */
			}
			xmax = 1.;
		    }
		    suma.r = 0., suma.i = 0.;
		    sumb.r = 0., sumb.i = 0.;

		    i__3 = j - 1;
		    for (jr = je; jr <= i__3; ++jr) {
			d_cnjg(&z__3, &s[jr + j * s_dim1]);
			i__4 = jr;
			z__2.r = z__3.r * work[i__4].r - z__3.i * work[i__4]
				.i, z__2.i = z__3.r * work[i__4].i + z__3.i * 
				work[i__4].r;
			z__1.r = suma.r + z__2.r, z__1.i = suma.i + z__2.i;
			suma.r = z__1.r, suma.i = z__1.i;
			d_cnjg(&z__3, &p[jr + j * p_dim1]);
			i__4 = jr;
			z__2.r = z__3.r * work[i__4].r - z__3.i * work[i__4]
				.i, z__2.i = z__3.r * work[i__4].i + z__3.i * 
				work[i__4].r;
			z__1.r = sumb.r + z__2.r, z__1.i = sumb.i + z__2.i;
			sumb.r = z__1.r, sumb.i = z__1.i;
/* L80: */
		    }
		    z__2.r = acoeff * suma.r, z__2.i = acoeff * suma.i;
		    d_cnjg(&z__4, &bcoeff);
		    z__3.r = z__4.r * sumb.r - z__4.i * sumb.i, z__3.i = 
			    z__4.r * sumb.i + z__4.i * sumb.r;
		    z__1.r = z__2.r - z__3.r, z__1.i = z__2.i - z__3.i;
		    sum.r = z__1.r, sum.i = z__1.i;

/*                 Form x(j) = - SUM / conjg( a*S(j,j) - b*P(j,j) ) */

/*                 with scaling and perturbation of the denominator */

		    i__3 = j + j * s_dim1;
		    z__3.r = acoeff * s[i__3].r, z__3.i = acoeff * s[i__3].i;
		    i__4 = j + j * p_dim1;
		    z__4.r = bcoeff.r * p[i__4].r - bcoeff.i * p[i__4].i, 
			    z__4.i = bcoeff.r * p[i__4].i + bcoeff.i * p[i__4]
			    .r;
		    z__2.r = z__3.r - z__4.r, z__2.i = z__3.i - z__4.i;
		    d_cnjg(&z__1, &z__2);
		    d__.r = z__1.r, d__.i = z__1.i;
		    if ((d__1 = d__.r, abs(d__1)) + (d__2 = d_imag(&d__), abs(
			    d__2)) <= dmin__) {
			z__1.r = dmin__, z__1.i = 0.;
			d__.r = z__1.r, d__.i = z__1.i;
		    }

		    if ((d__1 = d__.r, abs(d__1)) + (d__2 = d_imag(&d__), abs(
			    d__2)) < 1.) {
			if ((d__1 = sum.r, abs(d__1)) + (d__2 = d_imag(&sum), 
				abs(d__2)) >= bignum * ((d__3 = d__.r, abs(
				d__3)) + (d__4 = d_imag(&d__), abs(d__4)))) {
			    temp = 1. / ((d__1 = sum.r, abs(d__1)) + (d__2 = 
				    d_imag(&sum), abs(d__2)));
			    i__3 = j - 1;
			    for (jr = je; jr <= i__3; ++jr) {
				i__4 = jr;
				i__5 = jr;
				z__1.r = temp * work[i__5].r, z__1.i = temp * 
					work[i__5].i;
				work[i__4].r = z__1.r, work[i__4].i = z__1.i;
/* L90: */
			    }
			    xmax = temp * xmax;
			    z__1.r = temp * sum.r, z__1.i = temp * sum.i;
			    sum.r = z__1.r, sum.i = z__1.i;
			}
		    }
		    i__3 = j;
		    z__2.r = -sum.r, z__2.i = -sum.i;
		    wladiv_(&z__1, &z__2, &d__);
		    work[i__3].r = z__1.r, work[i__3].i = z__1.i;
/* Computing MAX */
		    i__3 = j;
		    d__3 = xmax, d__4 = (d__1 = work[i__3].r, abs(d__1)) + (
			    d__2 = d_imag(&work[j]), abs(d__2));
		    xmax = f2cmax(d__3,d__4);
/* L100: */
		}

/*              Back transform eigenvector if HOWMNY='B'. */

		if (ilback) {
		    i__2 = *n + 1 - je;
		    wgemv_("N", n, &i__2, &c_b2, &vl[je * vl_dim1 + 1], ldvl, 
			    &work[je], &c__1, &c_b1, &work[*n + 1], &c__1);
		    isrc = 2;
		    ibeg = 1;
		} else {
		    isrc = 1;
		    ibeg = je;
		}

/*              Copy and scale eigenvector into column of VL */

		xmax = 0.;
		i__2 = *n;
		for (jr = ibeg; jr <= i__2; ++jr) {
/* Computing MAX */
		    i__3 = (isrc - 1) * *n + jr;
		    d__3 = xmax, d__4 = (d__1 = work[i__3].r, abs(d__1)) + (
			    d__2 = d_imag(&work[(isrc - 1) * *n + jr]), abs(
			    d__2));
		    xmax = f2cmax(d__3,d__4);
/* L110: */
		}

		if (xmax > safmin) {
		    temp = 1. / xmax;
		    i__2 = *n;
		    for (jr = ibeg; jr <= i__2; ++jr) {
			i__3 = jr + ieig * vl_dim1;
			i__4 = (isrc - 1) * *n + jr;
			z__1.r = temp * work[i__4].r, z__1.i = temp * work[
				i__4].i;
			vl[i__3].r = z__1.r, vl[i__3].i = z__1.i;
/* L120: */
		    }
		} else {
		    ibeg = *n + 1;
		}

		i__2 = ibeg - 1;
		for (jr = 1; jr <= i__2; ++jr) {
		    i__3 = jr + ieig * vl_dim1;
		    vl[i__3].r = 0., vl[i__3].i = 0.;
/* L130: */
		}

	    }
L140:
	    ;
	}
    }

/*     Right eigenvectors */

    if (compr) {
	ieig = im + 1;

/*        Main loop over eigenvalues */

	for (je = *n; je >= 1; --je) {
	    if (ilall) {
		ilcomp = TRUE_;
	    } else {
		ilcomp = select[je];
	    }
	    if (ilcomp) {
		--ieig;

		i__1 = je + je * s_dim1;
		i__2 = je + je * p_dim1;
		if ((d__2 = s[i__1].r, abs(d__2)) + (d__3 = d_imag(&s[je + je 
			* s_dim1]), abs(d__3)) <= safmin && (d__1 = p[i__2].r,
			 abs(d__1)) <= safmin) {

/*                 Singular matrix pencil -- return unit eigenvector */

		    i__1 = *n;
		    for (jr = 1; jr <= i__1; ++jr) {
			i__2 = jr + ieig * vr_dim1;
			vr[i__2].r = 0., vr[i__2].i = 0.;
/* L150: */
		    }
		    i__1 = ieig + ieig * vr_dim1;
		    vr[i__1].r = 1., vr[i__1].i = 0.;
		    goto L250;
		}

/*              Non-singular eigenvalue: */
/*              Compute coefficients  a  and  b  in */

/*              ( a A - b B ) x  = 0 */

/* Computing MAX */
		i__1 = je + je * s_dim1;
		i__2 = je + je * p_dim1;
		d__4 = ((d__2 = s[i__1].r, abs(d__2)) + (d__3 = d_imag(&s[je 
			+ je * s_dim1]), abs(d__3))) * ascale, d__5 = (d__1 = 
			p[i__2].r, abs(d__1)) * bscale, d__4 = f2cmax(d__4,d__5);
		temp = 1. / f2cmax(d__4,safmin);
		i__1 = je + je * s_dim1;
		z__2.r = temp * s[i__1].r, z__2.i = temp * s[i__1].i;
		z__1.r = ascale * z__2.r, z__1.i = ascale * z__2.i;
		salpha.r = z__1.r, salpha.i = z__1.i;
		i__1 = je + je * p_dim1;
		sbeta = temp * p[i__1].r * bscale;
		acoeff = sbeta * ascale;
		z__1.r = bscale * salpha.r, z__1.i = bscale * salpha.i;
		bcoeff.r = z__1.r, bcoeff.i = z__1.i;

/*              Scale to avoid underflow */

		lsa = abs(sbeta) >= safmin && abs(acoeff) < small;
		lsb = (d__1 = salpha.r, abs(d__1)) + (d__2 = d_imag(&salpha), 
			abs(d__2)) >= safmin && (d__3 = bcoeff.r, abs(d__3)) 
			+ (d__4 = d_imag(&bcoeff), abs(d__4)) < small;

		scale = 1.;
		if (lsa) {
		    scale = small / abs(sbeta) * f2cmin(anorm,big);
		}
		if (lsb) {
/* Computing MAX */
		    d__3 = scale, d__4 = small / ((d__1 = salpha.r, abs(d__1))
			     + (d__2 = d_imag(&salpha), abs(d__2))) * f2cmin(
			    bnorm,big);
		    scale = f2cmax(d__3,d__4);
		}
		if (lsa || lsb) {
/* Computing MIN */
/* Computing MAX */
		    d__5 = 1., d__6 = abs(acoeff), d__5 = f2cmax(d__5,d__6), 
			    d__6 = (d__1 = bcoeff.r, abs(d__1)) + (d__2 = 
			    d_imag(&bcoeff), abs(d__2));
		    d__3 = scale, d__4 = 1. / (safmin * f2cmax(d__5,d__6));
		    scale = f2cmin(d__3,d__4);
		    if (lsa) {
			acoeff = ascale * (scale * sbeta);
		    } else {
			acoeff = scale * acoeff;
		    }
		    if (lsb) {
			z__2.r = scale * salpha.r, z__2.i = scale * salpha.i;
			z__1.r = bscale * z__2.r, z__1.i = bscale * z__2.i;
			bcoeff.r = z__1.r, bcoeff.i = z__1.i;
		    } else {
			z__1.r = scale * bcoeff.r, z__1.i = scale * bcoeff.i;
			bcoeff.r = z__1.r, bcoeff.i = z__1.i;
		    }
		}

		acoefa = abs(acoeff);
		bcoefa = (d__1 = bcoeff.r, abs(d__1)) + (d__2 = d_imag(&
			bcoeff), abs(d__2));
		xmax = 1.;
		i__1 = *n;
		for (jr = 1; jr <= i__1; ++jr) {
		    i__2 = jr;
		    work[i__2].r = 0., work[i__2].i = 0.;
/* L160: */
		}
		i__1 = je;
		work[i__1].r = 1., work[i__1].i = 0.;
/* Computing MAX */
		d__1 = ulp * acoefa * anorm, d__2 = ulp * bcoefa * bnorm, 
			d__1 = f2cmax(d__1,d__2);
		dmin__ = f2cmax(d__1,safmin);

/*              Triangular solve of  (a A - b B) x = 0  (columnwise) */

/*              WORK(1:j-1) contains sums w, */
/*              WORK(j+1:JE) contains x */

		i__1 = je - 1;
		for (jr = 1; jr <= i__1; ++jr) {
		    i__2 = jr;
		    i__3 = jr + je * s_dim1;
		    z__2.r = acoeff * s[i__3].r, z__2.i = acoeff * s[i__3].i;
		    i__4 = jr + je * p_dim1;
		    z__3.r = bcoeff.r * p[i__4].r - bcoeff.i * p[i__4].i, 
			    z__3.i = bcoeff.r * p[i__4].i + bcoeff.i * p[i__4]
			    .r;
		    z__1.r = z__2.r - z__3.r, z__1.i = z__2.i - z__3.i;
		    work[i__2].r = z__1.r, work[i__2].i = z__1.i;
/* L170: */
		}
		i__1 = je;
		work[i__1].r = 1., work[i__1].i = 0.;

		for (j = je - 1; j >= 1; --j) {

/*                 Form x(j) := - w(j) / d */
/*                 with scaling and perturbation of the denominator */

		    i__1 = j + j * s_dim1;
		    z__2.r = acoeff * s[i__1].r, z__2.i = acoeff * s[i__1].i;
		    i__2 = j + j * p_dim1;
		    z__3.r = bcoeff.r * p[i__2].r - bcoeff.i * p[i__2].i, 
			    z__3.i = bcoeff.r * p[i__2].i + bcoeff.i * p[i__2]
			    .r;
		    z__1.r = z__2.r - z__3.r, z__1.i = z__2.i - z__3.i;
		    d__.r = z__1.r, d__.i = z__1.i;
		    if ((d__1 = d__.r, abs(d__1)) + (d__2 = d_imag(&d__), abs(
			    d__2)) <= dmin__) {
			z__1.r = dmin__, z__1.i = 0.;
			d__.r = z__1.r, d__.i = z__1.i;
		    }

		    if ((d__1 = d__.r, abs(d__1)) + (d__2 = d_imag(&d__), abs(
			    d__2)) < 1.) {
			i__1 = j;
			if ((d__1 = work[i__1].r, abs(d__1)) + (d__2 = d_imag(
				&work[j]), abs(d__2)) >= bignum * ((d__3 = 
				d__.r, abs(d__3)) + (d__4 = d_imag(&d__), abs(
				d__4)))) {
			    i__1 = j;
			    temp = 1. / ((d__1 = work[i__1].r, abs(d__1)) + (
				    d__2 = d_imag(&work[j]), abs(d__2)));
			    i__1 = je;
			    for (jr = 1; jr <= i__1; ++jr) {
				i__2 = jr;
				i__3 = jr;
				z__1.r = temp * work[i__3].r, z__1.i = temp * 
					work[i__3].i;
				work[i__2].r = z__1.r, work[i__2].i = z__1.i;
/* L180: */
			    }
			}
		    }

		    i__1 = j;
		    i__2 = j;
		    z__2.r = -work[i__2].r, z__2.i = -work[i__2].i;
		    wladiv_(&z__1, &z__2, &d__);
		    work[i__1].r = z__1.r, work[i__1].i = z__1.i;

		    if (j > 1) {

/*                    w = w + x(j)*(a S(*,j) - b P(*,j) ) with scaling */

			i__1 = j;
			if ((d__1 = work[i__1].r, abs(d__1)) + (d__2 = d_imag(
				&work[j]), abs(d__2)) > 1.) {
			    i__1 = j;
			    temp = 1. / ((d__1 = work[i__1].r, abs(d__1)) + (
				    d__2 = d_imag(&work[j]), abs(d__2)));
			    if (acoefa * rwork[j] + bcoefa * rwork[*n + j] >= 
				    bignum * temp) {
				i__1 = je;
				for (jr = 1; jr <= i__1; ++jr) {
				    i__2 = jr;
				    i__3 = jr;
				    z__1.r = temp * work[i__3].r, z__1.i = 
					    temp * work[i__3].i;
				    work[i__2].r = z__1.r, work[i__2].i = 
					    z__1.i;
/* L190: */
				}
			    }
			}

			i__1 = j;
			z__1.r = acoeff * work[i__1].r, z__1.i = acoeff * 
				work[i__1].i;
			ca.r = z__1.r, ca.i = z__1.i;
			i__1 = j;
			z__1.r = bcoeff.r * work[i__1].r - bcoeff.i * work[
				i__1].i, z__1.i = bcoeff.r * work[i__1].i + 
				bcoeff.i * work[i__1].r;
			cb.r = z__1.r, cb.i = z__1.i;
			i__1 = j - 1;
			for (jr = 1; jr <= i__1; ++jr) {
			    i__2 = jr;
			    i__3 = jr;
			    i__4 = jr + j * s_dim1;
			    z__3.r = ca.r * s[i__4].r - ca.i * s[i__4].i, 
				    z__3.i = ca.r * s[i__4].i + ca.i * s[i__4]
				    .r;
			    z__2.r = work[i__3].r + z__3.r, z__2.i = work[
				    i__3].i + z__3.i;
			    i__5 = jr + j * p_dim1;
			    z__4.r = cb.r * p[i__5].r - cb.i * p[i__5].i, 
				    z__4.i = cb.r * p[i__5].i + cb.i * p[i__5]
				    .r;
			    z__1.r = z__2.r - z__4.r, z__1.i = z__2.i - 
				    z__4.i;
			    work[i__2].r = z__1.r, work[i__2].i = z__1.i;
/* L200: */
			}
		    }
/* L210: */
		}

/*              Back transform eigenvector if HOWMNY='B'. */

		if (ilback) {
		    wgemv_("N", n, &je, &c_b2, &vr[vr_offset], ldvr, &work[1],
			     &c__1, &c_b1, &work[*n + 1], &c__1);
		    isrc = 2;
		    iend = *n;
		} else {
		    isrc = 1;
		    iend = je;
		}

/*              Copy and scale eigenvector into column of VR */

		xmax = 0.;
		i__1 = iend;
		for (jr = 1; jr <= i__1; ++jr) {
/* Computing MAX */
		    i__2 = (isrc - 1) * *n + jr;
		    d__3 = xmax, d__4 = (d__1 = work[i__2].r, abs(d__1)) + (
			    d__2 = d_imag(&work[(isrc - 1) * *n + jr]), abs(
			    d__2));
		    xmax = f2cmax(d__3,d__4);
/* L220: */
		}

		if (xmax > safmin) {
		    temp = 1. / xmax;
		    i__1 = iend;
		    for (jr = 1; jr <= i__1; ++jr) {
			i__2 = jr + ieig * vr_dim1;
			i__3 = (isrc - 1) * *n + jr;
			z__1.r = temp * work[i__3].r, z__1.i = temp * work[
				i__3].i;
			vr[i__2].r = z__1.r, vr[i__2].i = z__1.i;
/* L230: */
		    }
		} else {
		    iend = 0;
		}

		i__1 = *n;
		for (jr = iend + 1; jr <= i__1; ++jr) {
		    i__2 = jr + ieig * vr_dim1;
		    vr[i__2].r = 0., vr[i__2].i = 0.;
/* L240: */
		}

	    }
L250:
	    ;
	}
    }

    return;

/*     End of ZTGEVC */

} /* wtgevc_ */

