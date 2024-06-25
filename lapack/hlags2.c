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

/* > \brief \b DLAGS2 computes 2-by-2 orthogonal matrices U, V, and Q, and applies them to matrices A and B su
ch that the rows of the transformed A and B are parallel. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DLAGS2 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlags2.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlags2.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlags2.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DLAGS2( UPPER, A1, A2, A3, B1, B2, B3, CSU, SNU, CSV, */
/*                          SNV, CSQ, SNQ ) */

/*       LOGICAL            UPPER */
/*       DOUBLE PRECISION   A1, A2, A3, B1, B2, B3, CSQ, CSU, CSV, SNQ, */
/*      $                   SNU, SNV */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > DLAGS2 computes 2-by-2 orthogonal matrices U, V and Q, such */
/* > that if ( UPPER ) then */
/* > */
/* >           U**T *A*Q = U**T *( A1 A2 )*Q = ( x  0  ) */
/* >                             ( 0  A3 )     ( x  x  ) */
/* > and */
/* >           V**T*B*Q = V**T *( B1 B2 )*Q = ( x  0  ) */
/* >                            ( 0  B3 )     ( x  x  ) */
/* > */
/* > or if ( .NOT.UPPER ) then */
/* > */
/* >           U**T *A*Q = U**T *( A1 0  )*Q = ( x  x  ) */
/* >                             ( A2 A3 )     ( 0  x  ) */
/* > and */
/* >           V**T*B*Q = V**T*( B1 0  )*Q = ( x  x  ) */
/* >                           ( B2 B3 )     ( 0  x  ) */
/* > */
/* > The rows of the transformed A and B are parallel, where */
/* > */
/* >   U = (  CSU  SNU ), V = (  CSV SNV ), Q = (  CSQ   SNQ ) */
/* >       ( -SNU  CSU )      ( -SNV CSV )      ( -SNQ   CSQ ) */
/* > */
/* > Z**T denotes the transpose of Z. */
/* > */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] UPPER */
/* > \verbatim */
/* >          UPPER is LOGICAL */
/* >          = .TRUE.: the input matrices A and B are upper triangular. */
/* >          = .FALSE.: the input matrices A and B are lower triangular. */
/* > \endverbatim */
/* > */
/* > \param[in] A1 */
/* > \verbatim */
/* >          A1 is DOUBLE PRECISION */
/* > \endverbatim */
/* > */
/* > \param[in] A2 */
/* > \verbatim */
/* >          A2 is DOUBLE PRECISION */
/* > \endverbatim */
/* > */
/* > \param[in] A3 */
/* > \verbatim */
/* >          A3 is DOUBLE PRECISION */
/* >          On entry, A1, A2 and A3 are elements of the input 2-by-2 */
/* >          upper (lower) triangular matrix A. */
/* > \endverbatim */
/* > */
/* > \param[in] B1 */
/* > \verbatim */
/* >          B1 is DOUBLE PRECISION */
/* > \endverbatim */
/* > */
/* > \param[in] B2 */
/* > \verbatim */
/* >          B2 is DOUBLE PRECISION */
/* > \endverbatim */
/* > */
/* > \param[in] B3 */
/* > \verbatim */
/* >          B3 is DOUBLE PRECISION */
/* >          On entry, B1, B2 and B3 are elements of the input 2-by-2 */
/* >          upper (lower) triangular matrix B. */
/* > \endverbatim */
/* > */
/* > \param[out] CSU */
/* > \verbatim */
/* >          CSU is DOUBLE PRECISION */
/* > \endverbatim */
/* > */
/* > \param[out] SNU */
/* > \verbatim */
/* >          SNU is DOUBLE PRECISION */
/* >          The desired orthogonal matrix U. */
/* > \endverbatim */
/* > */
/* > \param[out] CSV */
/* > \verbatim */
/* >          CSV is DOUBLE PRECISION */
/* > \endverbatim */
/* > */
/* > \param[out] SNV */
/* > \verbatim */
/* >          SNV is DOUBLE PRECISION */
/* >          The desired orthogonal matrix V. */
/* > \endverbatim */
/* > */
/* > \param[out] CSQ */
/* > \verbatim */
/* >          CSQ is DOUBLE PRECISION */
/* > \endverbatim */
/* > */
/* > \param[out] SNQ */
/* > \verbatim */
/* >          SNQ is DOUBLE PRECISION */
/* >          The desired orthogonal matrix Q. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup doubleOTHERauxiliary */

/*  ===================================================================== */
void  hlags2_(logical *upper, halfreal *a1, halfreal *a2, 
	halfreal *a3, halfreal *b1, halfreal *b2, halfreal *b3, 
	halfreal *csu, halfreal *snu, halfreal *csv, halfreal *snv, 
	halfreal *csq, halfreal *snq)
{
    /* System generated locals */
    halfreal d__1;

    /* Local variables */
    halfreal a, b, c__, d__, r__, s1, s2, ua11, ua12, ua21, ua22, vb11, 
	    vb12, vb21, vb22, csl, csr, snl, snr, aua11, aua12, aua21, aua22, 
	    avb11, avb12, avb21, avb22, ua11r, ua22r, vb11r, vb22r;
    extern void  hlasv2_(halfreal *, halfreal *, 
	    halfreal *, halfreal *, halfreal *, halfreal *, 
	    halfreal *, halfreal *, halfreal *), hlartg_(halfreal *, 
	    halfreal *, halfreal *, halfreal *, halfreal *);


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


    if (*upper) {

/*        Input matrices A and B are upper triangular matrices */

/*        Form matrix C = A*adj(B) = ( a b ) */
/*                                   ( 0 d ) */

	a = *a1 * *b3;
	d__ = *a3 * *b1;
	b = *a2 * *b1 - *a1 * *b2;

/*        The SVD of doublereal 2-by-2 triangular C */

/*         ( CSL -SNL )*( A B )*(  CSR  SNR ) = ( R 0 ) */
/*         ( SNL  CSL ) ( 0 D ) ( -SNR  CSR )   ( 0 T ) */

	hlasv2_(&a, &b, &d__, &s1, &s2, &snr, &csr, &snl, &csl);

	if (abs(csl) >= abs(snl) || abs(csr) >= abs(snr)) {

/*           Compute the (1,1) and (1,2) elements of U**T *A and V**T *B, */
/*           and (1,2) element of |U|**T *|A| and |V|**T *|B|. */

	    ua11r = csl * *a1;
	    ua12 = csl * *a2 + snl * *a3;

	    vb11r = csr * *b1;
	    vb12 = csr * *b2 + snr * *b3;

	    aua12 = abs(csl) * abs(*a2) + abs(snl) * abs(*a3);
	    avb12 = abs(csr) * abs(*b2) + abs(snr) * abs(*b3);

/*           zero (1,2) elements of U**T *A and V**T *B */

	    if (abs(ua11r) + abs(ua12) != 0.) {
		if (aua12 / (abs(ua11r) + abs(ua12)) <= avb12 / (abs(vb11r) + 
			abs(vb12))) {
		    d__1 = -ua11r;
		    hlartg_(&d__1, &ua12, csq, snq, &r__);
		} else {
		    d__1 = -vb11r;
		    hlartg_(&d__1, &vb12, csq, snq, &r__);
		}
	    } else {
		d__1 = -vb11r;
		hlartg_(&d__1, &vb12, csq, snq, &r__);
	    }

	    *csu = csl;
	    *snu = -snl;
	    *csv = csr;
	    *snv = -snr;

	} else {

/*           Compute the (2,1) and (2,2) elements of U**T *A and V**T *B, */
/*           and (2,2) element of |U|**T *|A| and |V|**T *|B|. */

	    ua21 = -snl * *a1;
	    ua22 = -snl * *a2 + csl * *a3;

	    vb21 = -snr * *b1;
	    vb22 = -snr * *b2 + csr * *b3;

	    aua22 = abs(snl) * abs(*a2) + abs(csl) * abs(*a3);
	    avb22 = abs(snr) * abs(*b2) + abs(csr) * abs(*b3);

/*           zero (2,2) elements of U**T*A and V**T*B, and then swap. */

	    if (abs(ua21) + abs(ua22) != 0.) {
		if (aua22 / (abs(ua21) + abs(ua22)) <= avb22 / (abs(vb21) + 
			abs(vb22))) {
		    d__1 = -ua21;
		    hlartg_(&d__1, &ua22, csq, snq, &r__);
		} else {
		    d__1 = -vb21;
		    hlartg_(&d__1, &vb22, csq, snq, &r__);
		}
	    } else {
		d__1 = -vb21;
		hlartg_(&d__1, &vb22, csq, snq, &r__);
	    }

	    *csu = snl;
	    *snu = csl;
	    *csv = snr;
	    *snv = csr;

	}

    } else {

/*        Input matrices A and B are lower triangular matrices */

/*        Form matrix C = A*adj(B) = ( a 0 ) */
/*                                   ( c d ) */

	a = *a1 * *b3;
	d__ = *a3 * *b1;
	c__ = *a2 * *b3 - *a3 * *b2;

/*        The SVD of doublereal 2-by-2 triangular C */

/*         ( CSL -SNL )*( A 0 )*(  CSR  SNR ) = ( R 0 ) */
/*         ( SNL  CSL ) ( C D ) ( -SNR  CSR )   ( 0 T ) */

	hlasv2_(&a, &c__, &d__, &s1, &s2, &snr, &csr, &snl, &csl);

	if (abs(csr) >= abs(snr) || abs(csl) >= abs(snl)) {

/*           Compute the (2,1) and (2,2) elements of U**T *A and V**T *B, */
/*           and (2,1) element of |U|**T *|A| and |V|**T *|B|. */

	    ua21 = -snr * *a1 + csr * *a2;
	    ua22r = csr * *a3;

	    vb21 = -snl * *b1 + csl * *b2;
	    vb22r = csl * *b3;

	    aua21 = abs(snr) * abs(*a1) + abs(csr) * abs(*a2);
	    avb21 = abs(snl) * abs(*b1) + abs(csl) * abs(*b2);

/*           zero (2,1) elements of U**T *A and V**T *B. */

	    if (abs(ua21) + abs(ua22r) != 0.) {
		if (aua21 / (abs(ua21) + abs(ua22r)) <= avb21 / (abs(vb21) + 
			abs(vb22r))) {
		    hlartg_(&ua22r, &ua21, csq, snq, &r__);
		} else {
		    hlartg_(&vb22r, &vb21, csq, snq, &r__);
		}
	    } else {
		hlartg_(&vb22r, &vb21, csq, snq, &r__);
	    }

	    *csu = csr;
	    *snu = -snr;
	    *csv = csl;
	    *snv = -snl;

	} else {

/*           Compute the (1,1) and (1,2) elements of U**T *A and V**T *B, */
/*           and (1,1) element of |U|**T *|A| and |V|**T *|B|. */

	    ua11 = csr * *a1 + snr * *a2;
	    ua12 = snr * *a3;

	    vb11 = csl * *b1 + snl * *b2;
	    vb12 = snl * *b3;

	    aua11 = abs(csr) * abs(*a1) + abs(snr) * abs(*a2);
	    avb11 = abs(csl) * abs(*b1) + abs(snl) * abs(*b2);

/*           zero (1,1) elements of U**T*A and V**T*B, and then swap. */

	    if (abs(ua11) + abs(ua12) != 0.) {
		if (aua11 / (abs(ua11) + abs(ua12)) <= avb11 / (abs(vb11) + 
			abs(vb12))) {
		    hlartg_(&ua12, &ua11, csq, snq, &r__);
		} else {
		    hlartg_(&vb12, &vb11, csq, snq, &r__);
		}
	    } else {
		hlartg_(&vb12, &vb11, csq, snq, &r__);
	    }

	    *csu = snr;
	    *snu = csr;
	    *csv = snl;
	    *snv = csl;

	}

    }

    return;

/*     End of DLAGS2 */

} /* hlags2_ */

