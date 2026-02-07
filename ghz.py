import cudaq
import timeit

# Define a quantum kernel that returns an integer
@cudaq.kernel
def ghz_kernel(qubit_count: int):
    # Allocate qubits
    qubits = cudaq.qvector(qubit_count)

    # Create GHZ state
    h(qubits[0])
    for i in range(1, qubit_count):
        x.ctrl(qubits[0], qubits[i])

    # Measure qubits in z-basis
    mz(qubits)

if __name__ == "__main__":
    #Uncomment different sections for different backends

    #CPU only backend
    cudaq.set_target('qpp-cpu')

    #Single GPU backend
    #cudaq.set_target('nvidia')

    #Multi-GPU backend
    #cudaq.set_target('nvidia', option='mgpu')
    #Run in 2 GPUs with "mpirun -np 2 python ghz.py > out &

    #Tensornet backend
    #cudaq.set_target('tensornet', option='fp32')

    #Tensornet-mps backend
    #cudaq.set_target('tensornet-mps', option='fp32')

    #Multi-QPU backend
    #cudaq.set_target('nvidia', option='mqpu')

    #Change qubit count
    qubit_count = 5
    
    #Draw circuit
    #print(cudaq.draw(ghz_kernel, qubit_count))
    
    #Sample the circuit
    start_time = timeit.default_timer()
    results = cudaq.sample(ghz_kernel, qubit_count, shots_count=1000)
    end_time = timeit.default_timer()
    total_time = end_time - start_time
    
    #Print results
    print(f"Results: {results}, time: {total_time}")
