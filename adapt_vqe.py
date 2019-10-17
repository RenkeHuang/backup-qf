import openfermion
from openfermion.ops import FermionOperator
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.utils import hermitian_conjugated, commutator, normal_ordered
import openfermionpsi4
import numpy as np 
import scipy
import qforte
from qforte.utils.exponentiate import exponentiate_single_term

def runPsi4(geometry, kwargs):
    """ Returns an updated MolecularData object

    Parameters
    ----------
    geometry: list
        A list of tuples giving the coordinates of each atom.
        An example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))]
        Distances in angstrom. Use atomic symbols to specify atoms.
    kwargs: dict
        A dictionary to set up psi4 calculation
        keys: basis, multiplicity, charge, description, run_scf, run_mp2, run_cisd, run_ccsd, run_fci

    """
    basis = kwargs.get('basis', 'sto-3g')
    multiplicity = kwargs.get('multiplicity', 1)
    charge = kwargs.get('charge', 0)
    description = kwargs.get('description', '')

    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity, charge, description)

    molecule = openfermionpsi4.run_psi4(molecule,
                        run_scf=kwargs.get('run_scf', 1),
                        run_mp2=kwargs.get('run_mp2', 0),
                        run_cisd=kwargs.get('run_cisd', 0),
                        run_ccsd=kwargs.get('run_ccsd', 0),
                        run_fci=kwargs.get('run_fci', 0))
    return molecule


class ADAPT_VQE:
    """ ADAPT_VQE class is initiaized with an instance of openfermion.hamiltonians.MolecularData

    Attributes
    ----------
    molecule: instance of openfermion.hamiltonians.MolecularData class.

    fermion_ops: list of tuples of indices of spin orbitals included in single and double excitation operators.
        [{(i,a)}, {(j,i,b,a)}]
    
    jw_ops: list of qforte.QuantumOperator instances (A_i), obtained by jordan-wigner encoding all fermionic singles and doubles excitation operators.
    
    jw_commutators: list of qforte.QuantumOperator instances (commutator [H, A_i])
    
    qubit_hamiltonian: instance of qforte.QuantumOperator class
    
    h_qubit_OF: instance of openfermion MolecularOperator class
    
    occ_idx_alpha: list of indices of qubits encoding occupied alpha spin orbitals (even indices)
    
    occ_idx_beta: list of indices of qubits encoding occupied beta spin orbitals (odd indices)
    
    vir_idx_alpha: list of indices of qubits encoding virtual alpha spin orbitals (even indices)
    
    vir_idx_beta: list of indices of qubits encoding virtual beta spin orbitals (odd indices)
    
    ansatz_ops: list of qforte.QuantumOperator instances which prepare the wavefunction ansatz
    
    ansatz_op_idx: list of "jw_ops-indices" of operators in ansatz_ops list
    
    energy: list of variational optimized energies in all ADAPT cycles before convergence

    """
    def __init__(self, molecule_instance):
        self.molecule = molecule_instance
        self.fermion_ops = []
        self.jw_ops = []
        self.jw_commutators = []
        self.qubit_hamiltonian, self.h_qubit_OF = self.get_qubit_hamitonian()

        self.occ_idx_alpha = [
            2*i for i in range(molecule_instance.get_n_alpha_electrons())]
        self.occ_idx_beta = [
            2*i+1 for i in range(molecule_instance.get_n_beta_electrons())]
        self.vir_idx_alpha = [2*a for a in range(
            self.molecule.get_n_alpha_electrons(), molecule_instance.n_orbitals)]
        self.vir_idx_beta = [2*a+1 for a in range(
            self.molecule.get_n_beta_electrons(), molecule_instance.n_orbitals)]
        
        self.ansatz_ops = []
        self.ansatz_op_idx = []
        self.ansatz_fermion_ops = []
        self.energy = []


    def get_qubit_hamitonian(self, docc_indices=None, active_orb_indices=None):
        """ Return Hamiltonian operator as a qforte.QuantumOperator instance

        Parameters
        ----------
        docc_indices: list, optional
            list of spatial orbital indices indicating which orbitals should be considered doubly occupied.
        active_orb_indices: list, optional
            list of spatial orbital indices indicating which orbitals should be considered active.

        """
        # "molecular_hamiltonian": instance of the MolecularOperator class
        
        molecular_hamiltonian = self.molecule.get_molecular_hamiltonian(
            occupied_indices=docc_indices, active_indices=active_orb_indices)
        h_fermion = normal_ordered(get_fermion_operator(molecular_hamiltonian))
        h_qubit_OF = jordan_wigner(h_fermion)
        h_qubit_qf = qforte.build_from_openfermion(h_qubit_OF)
        return h_qubit_qf, h_qubit_OF 

    @property
    def ref_state(self):
        """ Return a qforte.QuantumComputer instance which represents the Hartree-Fock state of the input molecule
        """
        hf_qc = qforte.QuantumComputer(self.molecule.n_qubits)
        hf_cir = qforte.QuantumCircuit()

        for i_a in self.occ_idx_alpha:
            X_ia = qforte.make_gate('X', i_a, i_a)
            hf_cir.add_gate(X_ia)

        for i_b in self.occ_idx_beta:
            X_ib = qforte.make_gate('X', i_b, i_b)
            hf_cir.add_gate(X_ib)

        hf_qc.apply_circuit(hf_cir)
        return hf_qc

    def build_operator_pool(self):
        """ Call this function to populate jw_ops list and jw_commutators list
        """
        # Singles
        def add_singles(occ_idx, vir_idx):
            for i in occ_idx:
                for a in vir_idx:
                    single = FermionOperator(((a, 1), (i, 0)))
                    # 1. build Fermion anti-Hermitian operator
                    single -= hermitian_conjugated(single)
                    # 2. JW transformation to qubit operator
                    jw_single = jordan_wigner(single)
                    h_single_commutator = commutator(self.h_qubit_OF, jw_single)
                    # 3. qforte.build_from_openfermion(OF_qubitop)
                    qf_jw_single = qforte.build_from_openfermion(jw_single)
                    qf_commutator = qforte.build_from_openfermion(h_single_commutator)

                    self.fermion_ops.append((i,a))
                    self.jw_ops.append(qf_jw_single)
                    self.jw_commutators.append(qf_commutator)

        add_singles(self.occ_idx_alpha, self.vir_idx_alpha)
        add_singles(self.occ_idx_beta, self.vir_idx_beta)
                
        # Doubles
        def add_doubles(occ_idx_pairs, vir_idx_pairs):
            for ji in occ_idx_pairs:
                for ba in vir_idx_pairs:
                    j, i = ji
                    b, a = ba

                    double = FermionOperator(F'{a}^ {b}^ {i} {j}')
                    double -= hermitian_conjugated(double)
                    jw_double = jordan_wigner(double)
                    h_double_commutator = commutator(self.h_qubit_OF, jw_double)

                    qf_jw_double = qforte.build_from_openfermion(jw_double)
                    qf_commutator = qforte.build_from_openfermion(
                    h_double_commutator)

                    self.fermion_ops.append((j,i,b,a))
                    self.jw_ops.append(qf_jw_double)
                    self.jw_commutators.append(qf_commutator)

        from itertools import combinations, product

        occ_a_pairs = list(combinations(self.occ_idx_alpha, 2))
        vir_a_pairs = list(combinations(self.vir_idx_alpha, 2))
        add_doubles(occ_a_pairs, vir_a_pairs)

        occ_b_pairs = list(combinations(self.occ_idx_beta, 2))
        vir_b_pairs = list(combinations(self.vir_idx_beta, 2))
        add_doubles(occ_b_pairs, vir_b_pairs)

        occ_ab_pairs = list(product(self.occ_idx_alpha, self.occ_idx_beta))
        vir_ab_pairs = list(product(self.vir_idx_alpha, self.vir_idx_beta))
        add_doubles(occ_ab_pairs, vir_ab_pairs)

    def get_ansatz_circuit(self, param_list, provide_op_list = None):
        """ Return a qforte.QuantumCircuit object parametrized by input param_list
        
        Parameters
        ----------
        param_list: list
            list of parameters [param_1,..., param_n]
            to prepare the circuit 'exp(param_n*A_n)...exp(param_1*A_1)'
        
        Returns
        -------
        param_circuit: instance of qforte.QuantumCircuit class
            the circuit to be applied on the reference state to get wavefunction ansatz. 
        
        """
        if provide_op_list == None:
            op_list = self.ansatz_ops.copy()
        else:
            op_list = provide_op_list

        param_circuit = qforte.QuantumCircuit()
        for i in range(len(param_list)):
            param_i = param_list[i]
            op = op_list[i]
            # exp_op is a circuit object
            exp_op = qforte.QuantumCircuit()

            for coeff, term in op.terms():
                factor = coeff*param_i
                #'exponentiate_single_term' function returns a tuple (exponential(Cir), 1.0)
                exp_term = exponentiate_single_term(factor, term)[0]
                exp_op.add_circuit(exp_term)

            param_circuit.add_circuit(exp_op)
        return param_circuit

    def compute_gradient(self, param_list, use_analytic_grad=True, atol=1e-6, step_size=1e-4, gvec_truc=False):
        """ Return a list of energy gradients for all operators in jw_ops list

        Parameters
        ----------
        param_list: list
            list of parameters [param_1,..., param_n]
            to prepare the circuit 'exp(param_n*A_n)...exp(param_1*A_1)'
        use_analytic_grad: boolean, optional
            if False, return numerical gradient, use the value of step_size for the step size
        atol: float, optional
            if the value of partial derivative is smaller than atol, set it to 0
        step_size: float, optional
            use this value to compute numerical gradient

        Returns
        -------
        gradients: list
            list of energy gradients (Absolute Value) w.r.t. all param_i in the jw_ops list
        """
        gradients = []

        param_circuit = self.get_ansatz_circuit(param_list)
        current_wfn = self.ref_state
        current_wfn.apply_circuit(param_circuit)

        if not gvec_truc and use_analytic_grad:
            for i in range(len(self.jw_ops)):
                commutator = self.jw_commutators[i]
                term_sum = 0.0
                for term_i in commutator.terms():
                    term_sum += term_i[0] * current_wfn.perfect_measure_circuit(term_i[1])
                term_sum = np.real(term_sum)

                if abs(term_sum) < atol:
                    term_sum = 0.0
                gradients.append(term_sum)

        elif not gvec_truc and not use_analytic_grad:
            """NOT work!"""
            """Numerical Gradient (E(ti+delta_ti)-E(ti-delta_ti)) / (2*delta_ti)"""
            print('    Computing numerical gradient......')
            for i in range(len(self.jw_ops)):
                # param_i in wfn ansatz
                if i in self.ansatz_op_idx:
                    amplitudes = param_list
                    idx_in_ansatz = self.ansatz_op_idx.index(i)
                    op_list = None
                # param_i not in wfn ansatz, append it
                else:
                    amplitudes = np.append(param_list, 0.0)
                    op_list = self.ansatz_ops.copy()
                    op_list.append(self.jw_ops[i])
                    idx_in_ansatz = len(amplitudes) - 1

                a1 = amplitudes.copy()
                a1[idx_in_ansatz] = amplitudes[idx_in_ansatz]+step_size
                # vqe1 = self.variational_optimizer(a1, provide_op_list=op_list, print_level=0)
                # E_i_plus = vqe1.fun
                E_i_plus = self.compute_expectation(a1, provide_op_list=op_list)
                
                a2 = amplitudes.copy()
                a2[idx_in_ansatz] = amplitudes[idx_in_ansatz]-step_size
                # vqe2 = self.variational_optimizer(a2, provide_op_list=op_list, print_level=0)
                # E_i_minus = vqe2.fun
                E_i_minus = self.compute_expectation(a2, provide_op_list=op_list)

                gradient_i = (E_i_plus - E_i_minus) / (2*step_size)
                if abs(gradient_i) < atol:
                    gradient_i = 0.0
                gradients.append(gradient_i)

        elif gvec_truc == True:
            """NOT work! CANNOT be passed as jac to scipy minimizer!"""
            for i in range(len(param_list)):
                idx_in_pool = self.ansatz_op_idx[i]
                commutator = self.jw_commutators[idx_in_pool]
                term_sum = 0.0
                for term_i in commutator.terms():
                    term_sum += term_i[0] * current_wfn.perfect_measure_circuit(term_i[1])
                term_sum = np.real(term_sum)
                gradients.append(term_sum)

        return gradients


    def compute_expectation(self, param_list, provide_op_list=None):
        """ Return <wfn|H|wfn>

        Parameters
        ----------
        param_list: list
            list of parameters [param_1,..., param_n]
            to prepare the circuit 'exp(param_n*A_n)...exp(param_1*A_1)'

        Returns
        -------
        expectation: float
            <HF|exp(-param_1*A_1)...exp(-param_n*A_n)*H*exp(param_n*A_n)...exp(param_1*A_1)|HF>
        """
        param_circuit = self.get_ansatz_circuit(param_list, provide_op_list=provide_op_list)
        current_wfn = self.ref_state
        current_wfn.apply_circuit(param_circuit)

        expectation = 0.0
        for h_i in self.qubit_hamiltonian.terms():
            expectation += h_i[0] * current_wfn.perfect_measure_circuit(h_i[1])

        expectation = np.real(expectation)
        return expectation

    def variational_optimizer(self, input_params, theta = 1e-3, provide_op_list=None, print_level=1):
        """ Return scipy.optimize.OptimizeResult object

        Parameters
        ----------
        input_params: list
            list of paramaters which prepares the wavefunction ansatz

        Returns
        -------
        result: a scipy.optimize.OptimizeResult object with important attributes:
                - result.x: (ndarray) optimized parameters
                - result.success: (bool) whether or not the optimizer exited successfully
                - result.fun: (float) minimized expectation value (energy)

                - result.message: (string) description of the cause of the termination
        """


        input_param_array = np.asarray(input_params)
        if provide_op_list == None:
            result = scipy.optimize.minimize(
                self.compute_expectation, input_param_array, method='BFGS',\
                jac=False, options={'gtol': theta})
        else:
            # For compute numerical gradient (E(ti+delta_ti)-E(ti-delta_ti)) / (2*delta_ti)
            result = scipy.optimize.minimize(
                self.compute_expectation, input_param_array, method='BFGS', \
                jac=False, args=(provide_op_list),
                options={'gtol': theta})

        if print_level:
            print('        ---------------- Start VQE ----------------')
            print('        Optimizing parameters to minimize energy...')
            print('        '+str(len(input_params)) +
                ' parameters optimized after ' + str(result.nit) + ' iterations')
            print('        ---------------- Finish VQE ----------------')

        return result

    def iterating_cycle(self, max_cycle= 100, gradient_norm_threshold = 1e-3, print_details = True):
        print(F'\n    molecule info: {self.molecule.description}')
        print('\n    ================> Start ADAPT-VQE <================\n')
        params = []

        for i in range(1, max_cycle+1):
            print(F'    ====== ADAPT cycle {i} ======\n')
            gradients_i = self.compute_gradient(params, atol=1e-6)
            # print(F'    ** Analytic gradient (cycle {i}): {gradients_i}\n')
            # if i > 1:
            #     assert np.allclose(np.abs(gradients_i), np.abs(grad_i_numerical), atol=2e-3)

            norm = np.linalg.norm(gradients_i)
            
            if norm <= gradient_norm_threshold:
                print(F'    Norm of gradient vector:  {norm:2.10f}  ......Converged! (norm <= {gradient_norm_threshold})\n')
                print('    =========> Finish ADAPT-VQE Successfully! <=========\n\n')
                if print_details:
                    print('\n--- Details ---')
                    print('#op in op pool = ', len(self.jw_ops))
                    print('#commutator = ', len(self.jw_commutators))
                    print('#ansatzOp = ', len(self.ansatz_ops))
                    print('#ansatzOpIdx = ', len(self.ansatz_op_idx))
                    print(F'ansatz = {self.ansatz_fermion_ops}')
                    print(F'Energy from 1st   ADAPT iteration: {self.energy[0]:.10f} Hartree')
                    print(F'Energy from final ADAPT iteration: {self.energy[-1]:.10f} Hartree')
                    print('\n--- reference ---')
                    print(F'psi4 HF  energy: {self.molecule.hf_energy:.10f} Hartree.')
                    print(F'psi4 FCI energy: {self.molecule.fci_energy:.10f} Hartree.')
                break
                
            else:
                print(F'    Norm of gradient vector:  {norm:.10f}  ...larger than {gradient_norm_threshold}, Continue...')
                
                print(F'    Select the operator with Largest modulus of partial derivative...')
                idx_of_max_grad_i = gradients_i.index( max(gradients_i, key=abs) )
                # idx_of_max_grad_i = np.where(gradients_i == max(gradients_i))[0][0]
                print(F'    Index of the selected operator: {idx_of_max_grad_i}')
                self.ansatz_op_idx.append(idx_of_max_grad_i)
                print(F'    Growing wavefunction ansatz...')
                print(F'    Indices(in op pool) of operators in ansatz = {self.ansatz_op_idx}')

                A_i = self.jw_ops[idx_of_max_grad_i]
                self.ansatz_ops.append(A_i)
                t_i = self.fermion_ops[idx_of_max_grad_i]
                self.ansatz_fermion_ops.append(t_i)

                """ Test: initial guess of newly-added parameters impacts Convergence!
                H4, R=0.8; param_i = 0.005 for all i, 14 cycles, 
                correct op order = [22, 20, 15, 13, 10, 25, 16, 19, 9, 8, 4, 3, 0, 7]   
                           param_i = 0.001 for all i, 16 cycles,
                                   [22, 15, 20, 13, 10, 25, 19, 16, 8, 9, 0, 7, 4, 3, 4, 25]
                           param_i = 0.0001, 18 cycles,
                                   [22, 15, 20, 13, 10, 25, 19, 16, 8, 9, 0, 7, 4, 3, 13, 4, 4, 4]
                """
                # if i < 4:
                #     param_i = 0.001
                #     print(
                #         F'\n  ##  gradients_{i}[{idx_of_max_grad_i}]: {gradients_i[idx_of_max_grad_i]}\n')
                #     print(F'        gradients_{i}[22]: {gradients_i[22]}\n')
                #     print(F'        gradients_{i}[20]: {gradients_i[20]}\n')
                #     print(F'        gradients_{i}[15]: {gradients_i[15]}\n')
                # else:
                #     break
                """ End Test """

                param_i = 0.005
                params = np.append(params, param_i)

                result = self.variational_optimizer(params, theta=gradient_norm_threshold)
                params = result.x

                energy_i = result.fun
                self.energy.append(energy_i)
                print('    Optimized Energy of ADAPT cycle {}: {:.10f}\n'.format(i,energy_i))

                # """ Compute numerical gradient, 
                #     Compare grad_i_numerical with gradients_i+1(analytical)"""
                # grad_i_numerical = self.compute_gradient(params, use_analytic_grad=False)
                # # print('    Compare it with the analytic gradient of next cycle......\n')
                # print(F'    ** Numerical gradient (cycle {i}): {grad_i_numerical}\n\n')
                
        else: 
            print(F'Warning: Norm of gradient vector NOT Converge in {max_cycle} cycles! (norm > {gradient_norm_threshold}) ')
            
            # with 


