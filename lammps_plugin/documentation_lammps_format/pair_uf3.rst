.. index:: pair_style uf3
.. index:: pair_style uf3/kk

pair_style uf3 command
======================

Accelerator Variants: *uf3/kk*

Syntax
""""""

.. code-block:: LAMMPS

    pair_style style BodyFlag NumAtomType

 * style = *uf3* or *uf3/kk*
 
   .. parsed-literal::

        BodyFlag = Indicates whether to calculate only 2-body or 2 and 3-body interactions. Possible values- 2 or 3
        NumAtomType = Number of atoms types in the simulation



Examples
""""""""

.. code-block:: LAMMPS

    pair_style uf3 3 1
    pair_coeff 1 1 Nb_Nb
    pair_coeff 1 1 1 Nb_Nb_Nb

    pair_style uf3 2 2
    pair_coeff 1 1 Nb_Nb
    pair_coeff 1 2 Nb_Sn
    pair_coeff 2 2 Sn_Sn

    pair_style uf3 3 2
    pair_coeff 1 * Nb_Nb
    pair_coeff 2 * Sn_Sn
    pair_style 1 * * Nb_Sn_Sn
    pair_style 2 * * Sn_Sn_Sn

Description
"""""""""""

The *uf3* style computes the :ref:`Ultra-Fast Force Fields (UF3) <Xie23>` potential, a machine-learning interatomic potential. In UF3, the total energy of the system is defined via two- and three-body interactions:

.. math::

    E & = \sum_{i,j} V_2(r_{ij}) + \sum_{i,j,k} V_3 (r_{ij},r_{ik},r_{jk})

    V_2(r_{ij}) & = \sum_{n=0}^K c_n B_n(r_{ij})

    V_3 (r_{ij},r_{ik},r_{jk}) & = \sum_{l=0}^K_l \sum_{m=0}^K_m \sum_{n=0}^K_n c_{l,m,n} B_l(r_{ij}) B_m(r_{ik}) B_n(r_{jk})

where :math:`V_2(r_{ij})` and :math:`V_3 (r_{ij},r_{ik},r_{jk})` are the two- and three-body interactions. For the two-body the summation is over all neighbours J and for the three-body the summation is over all neighbors J and K of atom I within a cutoff distance determined from the potential files. :math:`B_n(r_{ij})` are the cubic bspline basis, :math:`c_n` and :math:`c_{l,m,n}` are the machine-learned interaction parameters and :math:`K`, :math:`K_l`, :math:`K_m`, and :math:`K_n` denote the number of basis functions per spline or tensor spline dimension.


----------

.. _Xie23:

**(Xie23)** S. R. Xie, M. Rupp, and R. G. Hennig, "Ultra-fast interpretable machine-learning potentials", preprint arXiv:2110.00624v2 (2023)
