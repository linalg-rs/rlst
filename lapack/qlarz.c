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
static quadreal c_b5 = 1.;

/* > \brief \b DLARZ applies an elementary reflector (as returned by stzrzf) to a general matrix. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DLARZ + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlarz.f
"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlarz.f
"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlarz.f
"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DLARZ( SIDE, M, N, L, V, INCV, TAU, C, LDC, WORK ) */

/*       CHARACTER          SIDE */
/*       INTEGER            INCV, L, LDC, M, N */
/*       DOUBLE PRECISION   TAU */
/*       DOUBLE PRECISION   C( LDC, * ), V( * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DLARZ applies a doublereal elementary reflector H to a doublereal M-by-N */
/* > matrix C, from either the left or the right. H is represented in the */
/* > form */
/* > */
/* >       H = I - tau * v * v**T */
/* > */
/* > where tau is a doublereal scalar and v is a doublereal vector. */
/* > */
/* > If tau = 0, then H is taken to be the unit matrix. */
/* > */
/* > */
/* > H is a product of k elementary reflectors as returned by DTZRZF. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] SIDE */
/* > \verbatim */
/* >          SIDE is CHARACTER*1 */
/* >          = 'L': form  H * C */
/* >          = 'R': form  C * H */
/* > \endverbatim */
/* > */
/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of rows of the matrix C. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of columns of the matrix C. */
/* > \endverbatim */
/* > */
/* > \param[in] L */
/* > \verbatim */
/* >          L is INTEGER */
/* >          The number of entries of the vector V containing */
/* >          the meaningful part of the Householder vectors. */
/* >          If SIDE = 'L', M >= L >= 0, if SIDE = 'R', N >= L >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] V */
/* > \verbatim */
/* >          V is DOUBLE PRECISION array, dimension (1+(L-1)*abs(INCV)) */
/* >          The vector v in the representation of H as returned by */
/* >          DTZRZF. V is not used if TAU = 0. */
/* > \endverbatim */
/* > */
/* > \param[in] INCV */
/* > \verbatim */
/* >          INCV is INTEGER */
/* >          The increment between elements of v. INCV <> 0. */
/* > \endverbatim */
/* > */
/* > \param[in] TAU */
/* > \verbatim */
/* >          TAU is DOUBLE PRECISION */
/* >          The value tau in the representation of H. */
/* > \endverbatim */
/* > */
/* > \param[in,out] C */
/* > \verbatim */
/* >          C is DOUBLE PRECISION array, dimension (LDC,N) */
/* >          On entry, the M-by-N matrix C. */
/* >          On exit, C is overwritten by the matrix H * C if SIDE = 'L', */
/* >          or C * H if SIDE = 'R'. */
/* > \endverbatim */
/* > */
/* > \param[in] LDC */
/* > \verbatim */
/* >          LDC is INTEGER */
/* >          The leading dimension of the array C. LDC >= f2cmax(1,M). */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is DOUBLE PRECISION array, dimension */
/* >                         (N) if SIDE = 'L' */
/* >                      or (M) if SIDE = 'R' */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup doubleOTHERcomputational */

/* > \par Contributors: */
/*  ================== */
/* > */
/* >    A. Petitet, Computer Science Dept., Univ. of Tenn., Knoxville, USA */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
void  qlarz_(char *side, integer *m, integer *n, integer *l, 
	quadreal *v, integer *incv, quadreal *tau, quadreal *c__, 
	integer *ldc, quadreal *work)
{
    /* System generated locals */
    integer c_dim1, c_offset;
    quadreal d__1;

    /* Local variables */
    extern void  qger_(integer *, integer *, quadreal *, 
	    quadreal *, integer *, quadreal *, integer *, quadreal *, 
	    integer *);
    extern logical lsame_(char *, char *);
    extern void  qgemv_(char *, integer *, integer *, 
	    quadreal *, quadreal *, integer *, quadreal *, integer *, 
	    quadreal *, quadreal *, integer *), qcopy_(integer *, 
	    quadreal *, integer *, quadreal *, integer *), qaxpy_(integer 
	    *, quadreal *, quadreal *, integer *, quadreal *, integer *)
	    ;


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    --v;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    if (lsame_(side, "L")) {

/*        Form  H * C */

	if (*tau != 0.) {

/*           w( 1:n ) = C( 1, 1:n ) */

	    qcopy_(n, &c__[c_offset], ldc, &work[1], &c__1);

/*           w( 1:n ) = w( 1:n ) + C( m-l+1:m, 1:n )**T * v( 1:l ) */

	    qgemv_("Transpose", l, n, &c_b5, &c__[*m - *l + 1 + c_dim1], ldc, 
		    &v[1], incv, &c_b5, &work[1], &c__1);

/*           C( 1, 1:n ) = C( 1, 1:n ) - tau * w( 1:n ) */

	    d__1 = -(*tau);
	    qaxpy_(n, &d__1, &work[1], &c__1, &c__[c_offset], ldc);

/*           C( m-l+1:m, 1:n ) = C( m-l+1:m, 1:n ) - ... */
/*                               tau * v( 1:l ) * w( 1:n )**T */

	    d__1 = -(*tau);
	    qger_(l, n, &d__1, &v[1], incv, &work[1], &c__1, &c__[*m - *l + 1 
		    + c_dim1], ldc);
	}

    } else {

/*        Form  C * H */

	if (*tau != 0.) {

/*           w( 1:m ) = C( 1:m, 1 ) */

	    qcopy_(m, &c__[c_offset], &c__1, &work[1], &c__1);

/*           w( 1:m ) = w( 1:m ) + C( 1:m, n-l+1:n, 1:n ) * v( 1:l ) */

	    qgemv_("No transpose", m, l, &c_b5, &c__[(*n - *l + 1) * c_dim1 + 
		    1], ldc, &v[1], incv, &c_b5, &work[1], &c__1);

/*           C( 1:m, 1 ) = C( 1:m, 1 ) - tau * w( 1:m ) */

	    d__1 = -(*tau);
	    qaxpy_(m, &d__1, &work[1], &c__1, &c__[c_offset], &c__1);

/*           C( 1:m, n-l+1:n ) = C( 1:m, n-l+1:n ) - ... */
/*                               tau * w( 1:m ) * v( 1:l )**T */

	    d__1 = -(*tau);
	    qger_(m, l, &d__1, &work[1], &c__1, &v[1], incv, &c__[(*n - *l + 
		    1) * c_dim1 + 1], ldc);

	}

    }

    return;

/*     End of DLARZ */

} /* qlarz_ */

