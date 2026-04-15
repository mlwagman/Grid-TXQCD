/*************************************************************************************
TXQCD project: minimal Nf=2 EOFA DWF HMC driver for an 8^3 x 16 ensemble.

Derived from HMC/Mobius2p1fEOFA_F1.cc with strange RHMC and Hasenbusch preconditioning
removed. Two independent ExactOneFlavourRatioPseudoFermionAction instances implement
Nf=2 = det(D_m)^2 / det(D_PV)^2 with independent heatbath samples per flavor, as
suggested for positivity of the individual u/d determinants in TXQCD (notes Eq. 9).
*************************************************************************************/

#include "disable_examples_without_instantiations.h"
#ifdef ENABLE_FERMION_INSTANTIATIONS

#include <Grid/Grid.h>

int main(int argc, char **argv) {
  using namespace Grid;

  Grid_init(&argc, &argv);
  std::cout << GridLogMessage << "Grid threads: " << GridThread::GetThreads() << std::endl;

  typedef WilsonImplR         FermionImplPolicy;
  typedef MobiusFermionD      FermionAction;
  typedef MobiusEOFAFermionD  FermionEOFAAction;
  typedef typename FermionAction::FermionField FermionField;

  typedef GenericHMCRunner<ForceGradient> HMCWrapper;

  IntegratorParameters MD;
  MD.name    = "ForceGradient";
  MD.MDsteps = 6;
  MD.trajL   = 1.0;

  HMCparameters HMCparams;
  HMCparams.StartTrajectory   = 0;
  HMCparams.Trajectories      = 20;
  HMCparams.NoMetropolisUntil = 0;
  HMCparams.StartingType      = std::string("ColdStart");
  HMCparams.MD                = MD;
  HMCWrapper TheHMC(HMCparams);

  TheHMC.Resources.AddFourDimGrid("gauge");

  CheckpointerParameters CPparams;
  CPparams.config_prefix = "ckpoint_TXQCD2fEOFA_lat";
  CPparams.rng_prefix    = "ckpoint_TXQCD2fEOFA_rng";
  CPparams.saveInterval  = 1;
  CPparams.format        = "IEEE64BIG";
  TheHMC.Resources.LoadNerscCheckpointer(CPparams);

  RNGModuleParameters RNGpar;
  RNGpar.serial_seeds   = "1 2 3 4 5";
  RNGpar.parallel_seeds = "6 7 8 9 10";
  TheHMC.Resources.SetRNGSeeds(RNGpar);

  typedef PlaquetteMod<HMCWrapper::ImplPolicy> PlaqObs;
  TheHMC.Resources.AddObservable<PlaqObs>();

  // Physics
  const int Ls   = 16;
  Real beta      = 2.13;
  Real light_mass = 0.01;
  Real pv_mass    = 1.0;
  RealD M5 = 1.8;
  RealD b  = 1.0;
  RealD c  = 0.0; // plain DWF (Mobius with b=1, c=0)

  auto GridPtr   = TheHMC.Resources.GetCartesian();
  auto GridRBPtr = TheHMC.Resources.GetRBCartesian();
  auto FGrid     = SpaceTimeGrid::makeFiveDimGrid(Ls, GridPtr);
  auto FrbGrid   = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, GridPtr);

  IwasakiGaugeActionR GaugeAction(beta);

  // temporary gauge field: the HMC driver replaces it when reading/creating U
  LatticeGaugeField U(GridPtr);

  std::vector<Complex> boundary = {1, 1, 1, -1};
  FermionAction::ImplParams FParams(boundary);

  double ActionStop = 1e-10;
  double DerivStop  = 1e-8;
  int    MaxCG      = 30000;
  ConjugateGradient<FermionField> ActionCG(ActionStop, MaxCG);
  ConjugateGradient<FermionField> DerivCG (DerivStop,  MaxCG);

  // EOFA operator pair: L uses (m_f=m, m_b=PV, shift=0, pm=-1),
  //                    R uses (m_f=PV, m_b=m, shift=-1, pm=+1).
  // Constructor: MobiusEOFAFermion(U, FGrid, FrbGrid, UGrid, UrbGrid,
  //                                mq1, mq2, mq3, shift, pm, M5, b, c, implParams)
  // Following the strange setup in Mobius2p1fEOFA_F1.cc lines 288-291 with light_mass.
  FermionEOFAAction Light_Op_L(U, *FGrid, *FrbGrid, *GridPtr, *GridRBPtr,
                               light_mass, light_mass, pv_mass, 0.0, -1,
                               M5, b, c, FParams);
  FermionEOFAAction Light_Op_R(U, *FGrid, *FrbGrid, *GridPtr, *GridRBPtr,
                               pv_mass,    light_mass, pv_mass, -1.0, 1,
                               M5, b, c, FParams);

  OneFlavourRationalParams OFRp;
  OFRp.lo        = 0.1;
  OFRp.hi        = 25.0;
  OFRp.MaxIter   = 10000;
  OFRp.tolerance = 1.0e-9;
  OFRp.degree    = 12;
  OFRp.precision = 50;

  // Two independent EOFA pseudofermions for Nf=2 = det(D_m)^2/det(D_PV)^2.
  // Each draws its own random phi at refresh, giving an unbiased Nf=2 estimator.
  ExactOneFlavourRatioPseudoFermionAction<FermionImplPolicy>
    EOFA_u(Light_Op_L, Light_Op_R,
           ActionCG, ActionCG, ActionCG, DerivCG, DerivCG, OFRp, true);
  ExactOneFlavourRatioPseudoFermionAction<FermionImplPolicy>
    EOFA_d(Light_Op_L, Light_Op_R,
           ActionCG, ActionCG, ActionCG, DerivCG, DerivCG, OFRp, true);

  ActionLevel<HMCWrapper::Field> Level1(1);
  ActionLevel<HMCWrapper::Field> Level2(6);

  Level1.push_back(&EOFA_u);
  Level1.push_back(&EOFA_d);
  Level2.push_back(&GaugeAction);

  TheHMC.TheAction.push_back(Level1);
  TheHMC.TheAction.push_back(Level2);

  std::cout << GridLogMessage << " Action complete: Nf=2 EOFA (u,d) + Iwasaki gauge" << std::endl;

  TheHMC.ReadCommandLine(argc, argv);
  TheHMC.Run();

  Grid_finalize();
  return 0;
}

#endif
