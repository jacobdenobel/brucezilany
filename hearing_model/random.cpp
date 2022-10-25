#include "bruce2018.h"

std::mt19937 GENERATOR;

template<typename D>
std::vector<double> random(const size_t n, D& d) {
    std::vector<double> r(n);
    for (auto& ri : r)
        ri = d(GENERATOR);
    return r;
}

std::vector<double> rand(const size_t n) {
    static std::uniform_real_distribution<double> d(0, 1.0);
    return random(n, d);
}

double rand1() {
    static std::uniform_real_distribution<double> d(0, 1.0);
    return d(GENERATOR);
}

std::vector<double> randn(const size_t n) {
    static std::normal_distribution<double> d(0, 1.0);
    return random(n, d);
}


std::vector<double> ffgn(int N, const double tdres, const double Hinput, const int noiseType, const int mu)
{
    using namespace std::complex_literals;
    const int resamp = (int)std::ceil(1e-1 / tdres);
    const int nop = N;
    N = (int)std::max(10.0, std::ceil(N / resamp) + 1);

    double H = Hinput - 1;
    bool fBn = true;
    if (Hinput <= 1)
    {
        H = Hinput;
        fBn = false;
    }

    std::vector<double> y(N);
    if (H == 0.5)
    {
        y = randn(N);
    }
    else
    {
        static int Nfft = 0, Nlast = 0;
        static double Hlast = 0.0;
        static std::vector<double> Zmag;
        static CArray Z;

        if (Nlast == 0)
        {
            Nfft = static_cast<int>(std::pow(2, std::ceil(log2(2 * (N - 1)))));
            const size_t NfftHalf = std::round(Nfft / 2);
            CArray fftdata(Nfft);
            Zmag.resize(Nfft);
            Z.resize(Nfft);

            std::generate(std::begin(fftdata), std::end(fftdata),
                [NfftHalf, H, n = 0, reverse = false]() mutable
                {
                    if (n + 1 > NfftHalf) reverse = true;
                    double k = !reverse ? n++ : n--;
                    return 0.5 * (pow(k + 1, 2. * H) - (2.0 * pow(k, 2.0 * H)) + pow(abs(k - 1), 2. * H));
                });

            fft(fftdata);
            for (size_t i = 0; i < Zmag.size(); ++i) {
                if (fftdata[i].real() < 0.0) {
                    throw(std::runtime_error("FFT produced > 0"));
                }
                Zmag[i] = std::sqrt(fftdata[i].real());
            }
            Nlast = N;
            Hlast = H;
        }


        std::vector<double> zr1;
        std::vector<double> zr2;

        if (noiseType == 0) {
            // No rng
            zr1 = std::vector<double>(Nfft, 1.0);
            zr2 = std::vector<double>(Nfft, 1.0);
        }
        else if (noiseType == 1) {
            // Fixed matlab rng
            zr1 = {
                 0.539001198446002,-0.333146282212077, 0.758784275258885,-0.960019229100215,
                -2.010902387858044,-0.014145783976321, 0.014846193555120, 0.179719933210648,
                -2.035475594737959,-0.357587732438863, 0.317062418711363,-1.266378348690577,
                 1.038708704838524,-2.500059203501081,-1.252332731960022, 1.230339014018892,
                -0.504687908175280, 0.919640621536610,-0.234470350850954, 0.530697743839911,
                 0.660825091280324, 0.855468294638247,-0.994629072636940,-2.231455213644026,
                 0.318559022665053, 0.632957296094154,-0.151148210794462,-0.816060813871062,
                -1.014897009384865, 0.518977711821625,-0.059474326486106, 0.731639398082223
            };
            zr2 = {
                -0.638409626955796,-0.061701505688751,-0.218192062027145, 0.203235982652021,
                -0.098642410359283, 0.945333174032015,-0.801457072154293,-0.085099820744463,
                0.789397946964058, 1.226327097545239,-0.900142192575332, 0.424849252031244,
                -0.387098269639317, 1.170523150888439,-0.072882198808166,-1.612913245229722,
                -0.702699919458338,-0.283874347267996, 0.450432043543390,-0.259699095922555,
                0.409258053752079, 1.926425247717760,-0.945190729563938,-0.854589093975853,
                -0.219510861979715, 0.449824239893538, 0.257557798875416, 0.212844513926846,
                -0.087690563274934, 0.231624682299529,-0.563183338456413,-1.188876899529859
            };
        }
        else if (noiseType > 1) {
            if (noiseType == 2)
                GENERATOR.seed(42);
            zr1 = randn(Nfft);
            zr2 = randn(Nfft);
        }

        for (size_t i = 0; i < Nfft; i++) {
            Z[i] = Zmag.at(i) * (zr1.at(i) + 1i * zr2.at(i));
        }

        ifft(Z);

        for (size_t i = 0; i < N; i++) {
            y.at(i) = Z[i].real() * std::sqrt(Nfft);
        }
    }

    if (fBn) {
        int s = 0;
        for (size_t i = 0; i < N; i++) {
            y.at(i) += s;
            s = y.at(i);
        }
    }
    std::vector<double> y2;
    resample(resamp, 1, y, y2);
    y2.resize(nop);

    const double sigma = mu < .2 ? 1.0 : mu < 20 ? 10 : mu / 2.0;
    for (auto& yi : y2)
        yi *= sigma;
    return y2;
}
