#include <cuda/experimental/stf.cuh>
#include <cub/cub.cuh>

#include <math.h>
#include <fstream>
#include <iostream>
#include <omp.h>
using namespace std;
#define CHUNKSIZE 40  // chunk size of each thread to be used by dynamic/guided process type
#define THREAD_NUM 16

using namespace cuda::experimental::stf;

// Globals
int i, j, k, psi, q, qi;
const double c0 = 30.0;
// Particles
int nb, ni, bt;
double  V, V1, V2, space;
// Density
const double Rho = 1000;
// Grid
int r, c, n1, n2, row_fact, col_fact;
double dx, dy, L, H, l1, l2, h1, h2 ;
// Time
double t_sim, t_total, steps;

const int stencil_size = 9;

__device__
static const int stencil_x[stencil_size] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};

__device__
static const int stencil_y[stencil_size] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};

template <typename T>
void setVal(context &ctx, logical_data<slice<T>> a, T val) {
    ctx.parallel_for(a.shape(), a.write()).set_symbol("setVal")->*[val]__device__(size_t i, auto da) {
        da(i) = val;
    };
}

template <typename BinaryOp>
struct OpWrapper
{
  OpWrapper(BinaryOp _op)
      : op(mv(_op)) {};

  template <typename T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const
  {
    return op(a, b);
  }

  BinaryOp op;
};

template <typename Ctx, typename InT, typename OutT, typename BinaryOp>
void exclusive_scan(
  Ctx& ctx, logical_data<slice<InT>> in_data, logical_data<slice<OutT>> out_data, BinaryOp&& op, OutT init_val)
{
  size_t nitems = in_data.shape().size();

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveScan(
    d_temp_storage,
    temp_storage_bytes,
    (InT*) nullptr,
    (OutT*) nullptr,
    OpWrapper<BinaryOp>(op),
    init_val,
    in_data.shape().size(),
    0);

  auto ltemp = ctx.logical_data(shape_of<slice<char>>(temp_storage_bytes));

  ctx.task(in_data.read(), out_data.write(), ltemp.write())
      ->*[&op, init_val, nitems](cudaStream_t stream, auto d_in, auto d_out, auto d_temp) {
            size_t tmp_size = d_temp.size();
            cub::DeviceScan::ExclusiveScan(
              (void*) d_temp.data_handle(),
              tmp_size,
              (InT*) d_in.data_handle(),
              (OutT*) d_out.data_handle(),
              OpWrapper<BinaryOp>(op),
              init_val,
              nitems,
              stream);
          };
}


void print2file(context &ctx, logical_data<slice<double>> la, int rows, int cols, string path) {

    ctx.host_launch(la.read()).set_symbol("print2file")->*[=](auto a) {
        // Print pointer to file
        ofstream myfile (path, ios_base::app);
        for (int ii=0; ii<rows; ii++) {
            for (int jj=0; jj<cols; jj++) {
                myfile << a(ii*cols+jj) << " ";
            }
            myfile << endl;
        }
        myfile.close();
    };
}

__device__
bool stencilBoundaryCheck(int rows, int cols, int rb, int cb) {
    // Check if stencil indeces are inside the domain
    if (rows < 0 || cols < 0)
        return 1;
    else if (rows >= rb || cols >= cb)
        return 1;
    return 0;
}

__inline__ __host__ __device__
void indexFixSoft(int& ii, int& jj, int rb, int cb) {
    // Fix index to be inside the domain

    if (ii == -1) {ii = 0;}
    else if (ii == rb) {ii = rb-1;}
    if (jj == -1) {jj = 0;}
    else if (jj == cb) {jj = cb-1;}
}

__inline__ __device__
bool coordBoundaryCheck(double xx, double yy, int rb, int cb, double Dx, double Dy) {
    // Checks if point is outside the domain

    if (yy<0 || yy>=Dy*rb)
        return 1;

    else if (xx<0 || xx>=Dx*cb)
        return 1;

    return 0;
}

__inline__ __device__
double kernel(double dr, double ad, double h) {
    // Kernel function definition

    double o = abs(dr)/h;

	if (0 <= o && o <= 0.5) {
		return (ad/h)*(pow(5/2-o, 4) - 5*pow(3/2-o, 4) + 10*pow(0.5-o, 4));
	}
	else if (0.5 < o && o <= 1.5) {
		return (ad/h)*(pow(5/2-o, 4) - 5*pow(3/2-o, 4));
	}
	else if (1.5 < o && o <= 2.5) {
		return (ad/h)*(pow(5/2-o, 4));
	}
    return 0;
}

__inline__ __device__
double gradKernel(double dr, double ad, double h) {
    // Derivative of Kernel function

    double o = abs(dr)/h;

	if (0 <= o && o <= 0.5) {
		return (ad/h)*(-4/h*pow(5/2-o, 3) + 20/h*pow(3/2-o, 3) - 40/h*pow(0.5-o, 3));
	}
	else if (0.5 < o && o <= 1.5) {
		return (ad/h)*(-4/h*pow(5/2-o, 3) + 20/h*pow(3/2-o, 3));
	}
	else if (1.5 < o && o <= 2.5) {
		return (ad/h)*(-4/h*pow(5/2-o, 3));
	}
    return 0;
}

double randZeroToOne() {
    return rand() / (RAND_MAX + 1.);
}

__device__
bool circleCheck(double a, double b, double radius) {
    // Checks if point is within radius

    a = abs(a); b = abs(b);
    if (a + b <= radius)
        return 1;
    if (a > radius)
        return 0;
    if (b > radius)
        return 0;
    if (a*a + b*b <= radius*radius)
        return 1;
    else
        return 0;
}

__inline__ __device__
double viscTensor(double dvel, double difr, double mui, double muj, double rhoi, double rhoj, double deltaw, double h) {
    // Calculates viscocity tensor

    double eh = 0.01*h*h;

    return -16*(mui+muj)/(pow(difr, 2)*(rhoi+rhoj) + eh)*(dvel*difr)*deltaw;
}

void setPressure(context &ctx, logical_data<slice<double>> a, logical_data<slice<double>> rho) {
    // Calculates pressure
    double gamma = 7.0;

    ctx.parallel_for(a.shape(), a.write(), rho.read()).set_symbol("setPressure")->*[=]__device__(size_t jj, auto da, auto drho) {
        da(jj) = Rho*c0*c0/gamma*(pow(drho(jj)/Rho, gamma) - 1);
    };
}

__inline__ __device__
void setPressure2(slice<double> &a, double rrho, int cols) {
    // Calculates pressure
    double gamma = 7.0;

    a[cols] = Rho*c0*c0/gamma*(pow(rrho/Rho, gamma) - 1);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL); cout.tie(NULL);

    context ctx;

    // Print variables
    double fps = 30;
    double print_count = 0;

    // Particles
    const int n = 10000;

//    Rho = 1000;

    // Time-step settings
    const double dt = 0.0001;
    steps = 100000;
    //steps = 100;
    t_total = dt*steps;
    t_sim = 0.0;

    // Boundary thickness
    bt = 3;
    // Dimensions
    L = 20; H = 10;

    // Geometry initialisation parameters
    h1 = 0.2*H; l1 = L;
//    h2 = 0.3*H; l2 = 0.15*L;
    h2 = 0.5*H; l2 = 0.10*L;
    V1 = h1*l1; V2 = h2*l2;
    V = V1 + V2;
    double ratio = V1/V;
    n1 = ratio*n; n2 = n-n1;
    double factor = L*H/V;
    int nx = sqrt(L/H*n*factor + pow((L-H), 2)/(4*H*L)) - (L-H)/(2*H);
    int ny = n*factor/nx;
    dx = L/(nx-1);
    int nbh_side = (L/dx+1+2*bt);
    int nbh = bt*nbh_side*2;
    int nbv_side = bt;
    int nbv = (H/dx+1)*nbv_side*2;
    nb = nbh+nbv;

    const double M_pi = 3.14159265359;
    const double ad = 96*M_pi/1199;

    auto create_and_init = [&](){
        auto ld = ctx.logical_data(shape_of<slice<double>>(n+nb));
        ctx.parallel_for(ld.shape(), ld.write())->*[]__device__(size_t i, auto v) {
            v(i) = 0.0;
        };
        return ld;
    };

    auto lfx = create_and_init();
    auto lfy = create_and_init();
    auto lu = create_and_init();
    auto lv = create_and_init();
    auto lu_old = create_and_init();
    auto lv_old = create_and_init();
    auto lx = create_and_init();
    auto ly = create_and_init();
    auto lx_old = create_and_init();
    auto ly_old = create_and_init();
    auto lrho = create_and_init();
    auto lrho_old = create_and_init();
    auto ldrho = create_and_init();
    auto lp = create_and_init();

    setVal(ctx, lrho, Rho);
    setVal(ctx, lrho_old, Rho);
    setPressure(ctx, lp, lrho);

    // Kernel - Grid parameters
    const double h = sqrt(2)*dx;
    const double Dx = 2.5*h;
    const double Dy = Dx;
    const int d = (Dx/dx+1)*(Dx/dx+1); // max particle inside cell
    r = H/Dy+0.5; c = L/Dx+0.5;
    row_fact = Dy / (dx*(bt-1)) + 0;
    col_fact = Dx / (dx*(bt-1)) + 0;
    const int rb = r+2*row_fact;
    const int cb = c+2*col_fact;

    auto ljaret = ctx.logical_data(shape_of<slice<int>>(n + nb));
    setVal(ctx, ljaret, -1);

    // Number of particle in each cell
    auto lndeg = ctx.logical_data(shape_of<slice<int>>(rb*cb));

    // exclusive Prefix-sum of lndeg (used to find the location of each cell)
    auto lndeg_sum = ctx.logical_data(shape_of<slice<int>>(rb*cb));
    setVal(ctx, lndeg, 0);

    long long iter_cnt = 0;

    // Prints
    double dt_max = 0.1*h/c0;
    cout << "dt: " << dt << endl;
    cout << "dt_max " << dt_max << endl;
    cout << "dx: " << dx << endl;
    cout << "Dx: " << Dx << endl;
    cout << "Max particles in cell: " << d << endl;

    auto initialise = [=](context &ctx, logical_data<slice<double>> la, logical_data<slice<double>> lb, double l1, double l2) {
        // Initialise flow particles
        bool randomIni = 1;
        space = row_fact*Dx;
        ctx.host_launch(la.rw(), lb.rw()).set_symbol("initialise")->*[=](auto a, auto b){
            if (randomIni == 1) {
                // Structured particles
                int jj = 0; int ii = 0;
                for (int kk=0; kk<n1; kk++) {
                    if (jj*dx > l1) {ii++; jj = 0;}
                    a(kk) = space + jj*dx;
                    b(kk) = space + ii*dx;
                    jj++;
                }
                jj = 0; ii = 0;
                for (int kk=n1; kk<n; kk++) {
                    if (jj*dx > l2) {ii++; jj = 0;}
                    a(kk) = space + jj*dx;
                    b(kk) = space + dx/2 + h1 + dx + ii*dx;
                    jj++;
                }
            }
            else {
                // Random particles
                for (int kk = 0; kk<n; kk++) {
                    a(kk) = space + randZeroToOne()*L;
                    b(kk) = space + randZeroToOne()*H;
                }
            }
        };
    };


    auto initialiseBoundaries = [=](context &ctx, logical_data<slice<double>> lx, logical_data<slice<double>> ly, int nbh, int nbv, int nbh_side, int nbv_side) {
        // Initialise boundary particles
        ctx.host_launch(lx.rw(), ly.rw()).set_symbol("initialiseBoundaries")->*[=](auto x, auto y) {
            double space2 = row_fact*Dx-(bt-0)*dx;
            int jj = 0; int ii = 0;
            for (int kk=n; kk<n+nbh/2; kk++) {
                if (jj*dx > L+2*bt*dx) {ii++; jj = 0;}
                x(kk) = space2 + jj*dx;
                y(kk) = space2 + ii*dx;
                jj++;
            }
            jj = 0; ii = 0;
            for (int kk=n+nbh/2; kk<n+nbh; kk++) {
                if (jj*dx > L+2*bt*dx) {ii++; jj = 0;}
                x(kk) = space2 + jj*dx;
                y(kk) = space2 + dx + bt*dx + H + ii*dx;  // dx + ??
                jj++;
            }
            jj = 0; ii = 0;
            for (int kk=n+nbh; kk<n+nbh+nbv/2; kk++) {
                if (jj*dx > (bt-1)*dx) {ii++; jj = 0;}
                x(kk) = space2 + jj*dx;
                y(kk) = space2 + bt*dx + ii*dx;
                jj++;
            }
            jj = 0; ii = 0;
            for (int kk=n+nbh+nbv/2; kk<n+nbv+nbh; kk++) {
                if (jj*dx >= 2*bt*dx) {ii++; jj = 0;}
                x(kk) = space2 + dx + bt*dx + L + jj*dx;  // dx + ??
                y(kk) = space2 + bt*dx + ii*dx;
                jj++;
                if (jj%nbv_side == 0) {ii++; jj = 0;}
            }
        };
    };


    // Final initialisation
    initialise(ctx, lx, ly, l1, l2);
    initialise(ctx, lx_old, ly_old, l1, l2);
    initialiseBoundaries(ctx, lx, ly, nbh, nbv, nbh_side, nbv_side);

    // Fluid properties
    const double mass = Rho * dx * dx;
    // mu = 8.9e-4;
    const double mu = 0.001;
    const double gx = 0.0;
    const double gy = -9.81;

    // File initialisations
    string pathx = "x.txt";
    string pathy = "y.txt";

    auto fileIni = [=](string path, int size) {
        // Initialise file
        int no = 12;
        ofstream ofs;
        ofs.open(path, ofstream::out | ofstream::trunc);
        ofstream myfile (path, ios_base::app);
        myfile << dt << " " << steps << " " << t_total << " ";
        myfile << Dx << " " << Dy << " " << dx << " ";
        myfile << bt << " " << L << " " << H << " " << n << " " << nb << " " << n1 << " ";
        for (int jj=0; jj<size-no; jj++) {myfile << 0 << " ";}
        myfile << endl;
        ofs.close();
    };

    fileIni(pathx, n+nb);
    fileIni(pathy, n+nb);

    // Time-step loop
    while (t_sim < t_total) {

        // Calculate neighbors of each particle (Update every 4 steps)
        if (iter_cnt % 1 == 0) {
            setVal(ctx, lndeg, 0);  // initialise

            // Count particles in each cell
            ctx.parallel_for(lx.shape(), lx.read(), ly.read(), lndeg.rw()).set_symbol("neighbors")->*[=] __device__ (size_t ni, auto x, auto y, auto ndeg) {
                    // Skip leaked particles
                    if (x[ni] == 0 && y[ni] == 0 && ni < n) return;

                    // Calculate i, j 
                    int i = (rb-1) - (int)(y(ni)/Dy);
                    int j = x(ni)/Dx;
                    indexFixSoft(i, j, rb, cb);

                    // Multiple threads can add particles to the same cell
                    atomicAdd(&ndeg[i*cb+j], 1);
            };

            exclusive_scan(ctx, lndeg, lndeg_sum,
                              [] __device__(const int& a, const int& b) {
                                   return a + b;
                               }, 0);

            setVal(ctx, lndeg, 0);  // initialise again

            ctx.parallel_for(lx.shape(), lx.read(), ly.read(), ljaret.write(), lndeg.rw(), lndeg_sum.read()).set_symbol("neighbors")->*[=]__device__(size_t ni, auto x, auto y, auto jaret, auto ndeg, auto ndeg_sum) {
                    // Skip leaked particles
                    if (x[ni] == 0 && y[ni] == 0 && ni < n) return;

                    // Calculate i, j 
                    int i = (rb-1) - (int)(y(ni)/Dy);
                    int j = x(ni)/Dx;
                    indexFixSoft(i, j, rb, cb);

                    // Calculate particle index
                    int pos_in_cell = atomicAdd(&ndeg[i*cb+j], 1);

                    // Position in "jaret"
                    int q = ndeg_sum[i*cb+j] + pos_in_cell;

                    jaret(q) = ni;
            };
        }

        // Particle loop
        ctx.parallel_for(box(n), lx.rw(), ly.rw(), lu.read(), lv.read(),
                                 lrho.rw(), ldrho.rw(),
                                 ljaret.read(), lndeg.read(), lndeg_sum.read(),
                                 lp.read(),
                                 lfx.rw(), lfy.rw()).set_symbol("particle loop")
        ->*[=]__device__(size_t ni, auto x, auto y, auto u, auto v,
                                    auto rho, auto drho,
                                    auto jaret, auto ndeg, auto ndeg_sum,
                                    auto p,
                                    auto fx, auto fy) {
            // Skip leaked particles
            if (x[ni] == 0 && y[ni] == 0) return;

            int q, qi, i, j;
            double weight0 = 0;
            double weight1 = 0;
            double weightx, weighty, du, dv, dr0, dwr;

            // i, j of current particle
            const int i0 = (rb-1) - (int)(y[ni]/Dy);
            const int j0 = x[ni]/Dx;

            // Neighbor-stencil loop
            for (int psi=0; psi<stencil_size; psi++) {
                // i, j of neighbor
                i = i0 + stencil_x[psi];
                j = j0 + stencil_y[psi];

                bool out_of_bounds = stencilBoundaryCheck(i, j, rb, cb);
                if (out_of_bounds == 1) {continue;}

                // Find all particle neighbours @ stencil cell
                for (int k=0; k<ndeg[i*cb+j]; k++) {

                    // Index of current particle
                    q = ndeg_sum[i*cb+j] + k;

                    // Index of neighbor
                    qi = jaret[q];

                    // Skip same particle
                    if (q == qi) {continue;}

                    // Distances
                    double difx = x[qi] - x[ni];
                    double dify = y[qi] - y[ni];

                    // Check if neighbor is within range
                    if (circleCheck(difx, dify, Dx)) {
                        // Velocities
                        du = u[qi] - u[ni];
                        dv = v[qi] - v[ni];
                        // Magnitude
                        dr0 = sqrt(difx*difx + dify*dify);
                        // Kernel gradient
                        dwr = gradKernel(dr0, ad, h);
                        // Weights
                        weightx = dwr*difx/(dr0+10e-20);
                        weighty = dwr*dify/(dr0+10e-20);

                        // Viscocities
                        double mui = mu;
                        double muj = mu;
                        double visc_termx = viscTensor(du, dr0, mui, muj, rho[ni], rho[qi], dwr, h);
                        double visc_termy = viscTensor(dv, dr0, mui, muj, rho[ni], rho[qi], dwr, h);

                        // Calculate sumation terms
                        double sum_termx = mass*((p[ni]/rho[ni]/rho[ni] + p[qi]/rho[qi]/rho[qi])*weightx + visc_termx);
                        double sum_termy = mass*((p[ni]/rho[ni]/rho[ni] + p[qi]/rho[qi]/rho[qi])*weighty + visc_termy);

                        // Calculate forces
                        fx[ni] += sum_termx;
                        fy[ni] += sum_termy;

                        // Update densities
                        drho[ni] += mass*(du*weightx + dv*weighty);

                        // Update boundary densities
                        if (qi > n) {
                            rho[qi] += mass*(du*weightx + dv*weighty)*dt;
                        }

                        // Smoothing technique sums
                        if (iter_cnt % 10 == 0) {
                            weight0 += kernel(dr0, ad, h);
                            weight1 += kernel(dr0, ad, h)/rho[qi];
                        }
                    }
                } // cell neighbors loop
            } // stencil neighbors loop

            // Smoothing technique
            if (iter_cnt % 10 == 0) {rho[ni] =  weight0/weight1;}

        }; // particle ni loop

//        double maxrho = -1.0;
        auto lmaxrho = ctx.logical_data(shape_of<scalar_view<double>>());
        // Time-integration update
        ctx.parallel_for(lrho.shape(), lrho.rw(), lrho_old.rw(), ldrho.rw(), lp.write(), lfx.write(), lx.write(), lfy.write(), ly.write(), lu.write(), lv.write(), lu_old.rw(), lv_old.rw(), lx_old.rw(), ly_old.rw(), lmaxrho.reduce(reducer::maxval<double>{})).set_symbol("update")->*[=]__device__(size_t ni, auto rho, auto rho_old, auto drho, auto p, auto fx, auto x, auto fy, auto y, auto u, auto v, auto u_old, auto v_old, auto x_old, auto y_old, auto &maxrho) {
            if (ni < n) { // exclude particles in the boundary
                // Particle densities
                rho[ni] = rho_old[ni] + drho[ni]*dt;
                double perc = abs(rho[ni]-rho_old[ni])/rho_old[ni]*100;
                if (perc > maxrho) {maxrho = perc;}
                rho_old[ni] = rho[ni];
                drho[ni] = 0;

                // Particle accelerations
                double ax = fx[ni] + gx;
                double ay = fy[ni] + gy;
                fx[ni] = 0;
                fy[ni] = 0;

                // Particle velocities
                u[ni] = u_old[ni] + ax*dt;
                v[ni] = v_old[ni] + ay*dt;

                // Particle positions
                x[ni] = x_old[ni] + u[ni]*dt;
                y[ni] = y_old[ni] + v[ni]*dt;

                // Old value updates
                u_old[ni] = u[ni];
                v_old[ni] = v[ni];
                x_old[ni] = x[ni];
                y_old[ni] = y[ni];

                // Check for leaked particles
                if (coordBoundaryCheck(x[ni], y[ni], rb, cb, Dx, Dy) == 1) {
                    x[ni] = 0; y[ni] = 0;  // place particle at [0, 0]
                }
            }

            setPressure2(p, rho[ni], ni);

        }; // particle ni loop

        // // Update pressures
        // Print x, y to files
        if (t_sim/(1/fps) > print_count) {
            print2file(ctx, lx, 1, n+nb, pathx);
            print2file(ctx, ly, 1, n+nb, pathy);
            print_count = print_count + 1;
            cout << "Time elapsed (%): " << t_sim/t_total*100 << endl;
            cout << "Rho change: " << ctx.wait(lmaxrho) << endl;
        }

        // Update elapsed time
        t_sim = t_sim + dt;
        iter_cnt++;  // iteration counter

    } // time-step loop

    ctx.finalize();

    return 0;
}
