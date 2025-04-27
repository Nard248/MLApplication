import numpy as np
from tqdm import tqdm


class Simulation:
    """
    A class to simulate a dynamical system involving nonlinear operations on a spatial grid using Fourier transforms.

    Attributes:
        dt, dx, dy (float): Temporal and spatial step sizes.
        Nx, Ny, Nt (int): Number of grid points in the x, y dimensions and time steps.
        myu_size (tuple): Shape of the small-scale noise matrix.
        myu_mstd (tuple): Mean and standard deviation for the normal distribution of the noise.
        exponent (ndarray): Precomputed exponential term used in the Fourier transform.
        step1 (ndarray): Precomputed term used in the nonlinear transformation step.
    """

    def __init__(self, d=(0.01, 100/320, 100/320), N=(100, 320, 320), myu_size=(10, 2, 2), myu_mstd=(5.4, 0.8)):
        """
        Initializes the simulation parameters and precomputes necessary Fourier transform terms.

        Args:
            d (tuple): Time step and spatial step sizes (dt, dx, dy).
            N (tuple): Number of time steps and spatial grid points (Nt, Nx, Ny).
            myu_size (tuple): Dimensions of the small-scale initial condition matrix.
            myu_mstd (tuple): Mean and standard deviation for the noise distribution (mean, std).
        """
        self.dt, self.dx, self.dy = d
        self.Nt, self.Nx, self.Ny = N
        self.myu_size = myu_size
        self.myu_mstd = myu_mstd
        
        kx = np.fft.fftfreq(self.Nx, self.dx)
        ky = np.fft.fftfreq(self.Ny, self.dy)
        Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
        q = 10**-6 - 4.0 * np.pi**2 * (Kx**2 + Ky**2)
        self.exponent = np.exp(q * self.dt)
        expm1 = np.expm1(q * self.dt)
        self.q = q
        self.step1 = expm1 / q
        self.step2 = (expm1 - self.dt*q) / (self.dt*(q**2))


    def non_linear_function(self, xx, yy):
        """Defines the nonlinear interaction in the system."""
        return xx * (yy - np.abs(xx)**2)


    def next_state(self, A, myu, order = 2):
        """Computes the next state of the system using the nonlinear transformation."""
        A_hat = np.fft.fft2(A)
        N_hat = np.fft.fft2(self.non_linear_function(A, myu))

        if order == 1 or "N_hat_prev" not in dir(self):
            self.N_hat_prev = N_hat
            return np.fft.ifft2(A_hat*self.exponent + N_hat*self.step1) 

        R = np.fft.ifft2(A_hat*self.exponent + N_hat*self.step1 - (N_hat - self.N_hat_prev)*self.step2*.01) 
        self.N_hat_prev = N_hat
        return R
    

    def compute_myu(self):
        """Generates a spatially extended noise matrix based on small-scale fluctuations."""
        myu_small = np.random.normal(*self.myu_mstd, size=self.myu_size)
        myu_small = np.abs(myu_small)

        scale = np.array((self.Nt, self.Nx, self.Ny)) // np.array(self.myu_size)
        myu = np.kron(myu_small, np.ones(scale))
        return myu


    def compute_state(self, myu):
        """Simulates the system over time, generating the state matrix."""
        A_0 = np.random.normal(size=(self.Nx, self.Ny)) * 0.01 + \
              np.random.normal(size=(self.Nx, self.Ny)) * 0.01j
        A = np.zeros([self.Nt, self.Nx, self.Ny], dtype=np.complex64)
        A[0] = A_0
        
        for i in tqdm(range(1, self.Nt), desc="Computing States"):
            A[i] = self.next_state(A[i - 1], myu[i - 1])
        return A

    
    def compute(self):
        """Main method to compute the simulation."""
        myu = self.compute_myu()
        state = self.compute_state(myu)
        return state, myu


    def check_properties(self, A, myu):
        """Prints properties of the matrices A and myu to ensure correctness and stability."""
        print("Unique Myus count\t", np.count_nonzero(np.unique(myu)))
        unique_values, counts = np.unique(myu, return_counts=True)
        print("Max value of myu:\t", np.max(myu))
        print("Min value of myu:\t", np.min(myu))
        print("Unique values:", unique_values.tolist())
        print("Counts:\t\t", counts)
        print(f"A.shape={A.shape},\nMyu.shape={myu.shape},\n")
        print("Any NaN values in Myu\t\t", np.isnan(myu).any())
        print("Any NaN values in A\t\t", np.isnan(A).any())
