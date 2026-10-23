#!/usr/bin/env python
"""Builds paper.tex from the real result files of Step 2 and Step 3. No number is typed by hand.
    python make_paper_final.py --step2 step2_results --step3 step3_results --out paper.tex
EDIT THE AUTHOR BLOCK BEFORE SUBMISSION."""
import argparse, json, os, re, shutil
import numpy as np, pandas as pd

AUTHOR = "YOUR NAME"
AFFIL = "Department of Chemistry, YOUR UNIVERSITY, COUNTRY"
EMAIL = "your.email@example.edu"
SUPERVISOR = ""        # e.g. "Prof. A. B. Name, Department of Materials Science, University"
CODE_URL = "https://github.com/YOUR-USERNAME/earth-abundant-bandgap-screening"

def tx(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))
def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"
def pc(x): return f"{100*x:.1f}\\%"

TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
def key(f):
    c = {}
    for el, n in TOKEN.findall(f): c[el] = c.get(el, 0.0) + (float(n) if n else 1.0)
    t = sum(c.values()); return tuple(sorted((e, round(n / t, 5)) for e, n in c.items()))
def fam(f):
    els = {e for e, _ in TOKEN.findall(f)}
    if "O" in els: return "oxide"
    if els & {"S", "Se", "Te"}: return "chalcogenide"
    if els & {"F", "Cl", "Br", "I"}: return "halide"
    if els & {"N", "P", "As", "Sb"}: return "pnictide"
    return "other"

ABUND = {"O":461000,"Si":282000,"Al":82300,"Fe":56300,"Ca":41500,"Na":23600,"Mg":23300,"K":20900,"Ti":5650,"P":1050,"Mn":950,
 "S":350,"Cl":145,"V":120,"Cr":102,"Zn":70,"Ni":84,"Cu":60,"Sn":2.3,"N":19,"Li":20,"Zr":165,"Ba":425,"Sr":370,"F":585,"Br":2.4}

def build(s2, s3, s4):
    P = pd.read_csv(f"{s2}/protocol_comparison.csv").set_index("protocol")
    FH = pd.read_csv(f"{s2}/family_holdout.csv").set_index("held_out_family")
    CAND = pd.read_csv(f"{s2}/all_candidates.csv", keep_default_na=False, na_values=[""])  # "NaN" is sodium nitride, not a missing value
    S2 = json.load(open(f"{s2}/summary.json"))
    M = pd.read_csv(f"{s3}/main_results.csv").set_index("metric")
    FAM = pd.read_csv(f"{s3}/per_family_system_protocol.csv").set_index("family")
    S3 = json.load(open(f"{s3}/summary.json"))
    LIT = pd.read_csv(f"{s3}/literature_validation_results.csv", keep_default_na=False, na_values=[""]) if os.path.exists(f"{s3}/literature_validation_results.csv") else pd.DataFrame()
    LV = S3.get("literature") or {}
    NL = int(LV.get("n_materials", 0))

    def mv(name): r = M.loc[name]; return r.value, r.ci_low, r.ci_high
    sa, sa_lo, sa_hi = mv("MAE all compounds - single-stage (eV)")
    ta, ta_lo, ta_hi = mv("MAE all compounds - two-stage hard rule (eV)")
    dd, dd_lo, dd_hi = mv("MAE difference single minus two-stage (eV)")
    sn, sn_lo, sn_hi = mv("MAE true non-metals - single-stage (eV)")
    tn, tn_lo, tn_hi = mv("MAE true non-metals - two-stage hard rule (eV)")
    rn, rn_lo, rn_hi = mv("MAE true non-metals - regressor if non-metal known (eV)")
    ac, ac_lo, ac_hi = mv("Metal vs non-metal accuracy")
    auc = S3["auc"]; q90 = S3["error90_nonmetals_eV"]
    nsys = S2["n_systems"]; nds = S2["n_dataset"]; nmshare = S2["nonmetal_share"]

    cand_n = len(CAND); cand30 = int((CAND.p_in_window >= 0.30).sum()); pmax = CAND.p_in_window.max()
    cf = CAND.groupby("family").agg(n=("formula", "size"), n30=("p_in_window", lambda s: int((s >= .30).sum())),
                                    pmax=("p_in_window", "max"))
    stats = dict(family=FAM, FH=FH)
    chal, pnic, oxid, hal = (FAM.loc[k] for k in ["chalcogenide", "pnictide", "oxide", "halide"])
    pf = P.loc["family"]; ps = P.loc["system"]; pr = P.loc["random"]

    # ---------------- tables ----------------
    def row(*c): return " & ".join(str(x) for x in c) + " \\\\"
    T_main = "\n".join([
        row("MAE, all compounds, single-stage", f"{f3(sa)} [{f3(sa_lo)}, {f3(sa_hi)}]"),
        row("MAE, all compounds, two-stage", f"{f3(ta)} [{f3(ta_lo)}, {f3(ta_hi)}]"),
        row("Difference (single $-$ two-stage)", f"{f3(dd)} [{f3(dd_lo)}, {f3(dd_hi)}]"),
        row("MAE, true non-metals, single-stage", f"{f3(sn)} [{f3(sn_lo)}, {f3(sn_hi)}]"),
        row("MAE, true non-metals, two-stage", f"{f3(tn)} [{f3(tn_lo)}, {f3(tn_hi)}]"),
        row("MAE, true non-metals, regressor given non-metal", f"{f3(rn)} [{f3(rn_lo)}, {f3(rn_hi)}]"),
        row("Metal/non-metal accuracy", f"{f3(ac)} [{f3(ac_lo)}, {f3(ac_hi)}]")])
    T_fam = []
    for k in ["chalcogenide", "pnictide", "oxide", "halide", "other"]:
        r = FAM.loc[k]
        reg = f"{f3(r.regressor_MAE)} [{f3(r.ci_low)}, {f3(r.ci_high)}]" if pd.notna(r.regressor_MAE) else "--"
        two = f3(r.two_stage_MAE_nonmetals) if pd.notna(r.two_stage_MAE_nonmetals) else "--"
        T_fam.append(row(tx(k), int(r.n), f2(r.nonmetal_share), f3(r.accuracy), reg, two))
    T_fam = "\n".join(T_fam)
    T_prot = "\n".join(row(k, int(P.loc[k].n_tested), f3(P.loc[k].MAE_single_all), f3(P.loc[k].MAE_two_stage_all),
                           f3(P.loc[k].MAE_single_nonmetals), f3(P.loc[k].MAE_two_stage_nonmetals),
                           f3(P.loc[k].MAE_regressor_if_nonmetal_known), f3(P.loc[k].metal_vs_nonmetal_accuracy),
                           f3(P.loc[k].metal_vs_nonmetal_AUC)) for k in ["random", "system", "family"])
    T_fh = "\n".join(row(tx(k), int(FH.loc[k].n), int(FH.loc[k].n_nonmetal), f3(FH.loc[k].MAE_regressor_nonmetals),
                         f3(FH.loc[k].metal_vs_nonmetal_accuracy)) for k in ["chalcogenide", "pnictide", "oxide", "halide"])
    T_cf = "\n".join(row(tx(k), int(cf.loc[k].n), int(cf.loc[k].n30), f2(cf.loc[k].pmax)) for k in cf.index)
    valkeys = {key(f): f for f in LIT.formula} if len(LIT) else {}
    top = CAND[CAND.family.isin(["chalcogenide", "pnictide"])].sort_values("p_in_window", ascending=False).head(12)
    T_top = "\n".join(row(tx(r.formula) + ("$^\\dagger$" if key(r.formula) in valkeys else ""), tx(r.family), f2(r.pred_gap_eV),
                          f2(r.p_nonmetal), f2(r.p_in_window), f2(r.e_form)) for r in top.itertuples())
    T_lit = ""
    if len(LIT):
        T_lit = "\n".join(row(tx(r.formula), f2(r.measured_mean), f2(r.pred_gap_eV), f2(r.p_nonmetal), f2(r.p_in_window),
                              f2(r.abs_error_eV)) for r in LIT.itertuples())

    CS = CAND.sort_values("p_in_window", ascending=False).reset_index(drop=True)
    CS["rank"] = CS.index + 1
    CS["k"] = CS.formula.apply(key)
    found = [CS[CS.k == key(f)].iloc[0] for f in LIT.formula] if len(LIT) else []
    found = [r for r in found if r is not None]
    maxrank = int(max(r["rank"] for r in found)) if found else 0
    sys_in = int(sum(bool(r.system_in_training) for r in found)) if found else 0
    ox8 = CAND[CAND.family == "oxide"].sort_values("p_in_window", ascending=False).head(8)
    ox_crmn = int(ox8.formula.apply(lambda f: any(e in {x for x, _ in TOKEN.findall(f)} for e in ("Cr", "Mn"))).sum())
    ox_line = (f"the eight highest-ranked oxides ({"all" if ox_crmn == 8 else str(ox_crmn) + " of them"} chromium or manganese oxides) had non-metal probabilities of only "
               f"{f2(ox8.p_nonmetal.min())}--{f2(ox8.p_nonmetal.max())}")
    ab = sorted(ABUND.items(), key=lambda kv: -kv[1])
    cells = [f"{e} & {v:g}" for e, v in ab]
    while len(cells) % 4: cells.append(" & ")
    T_abund = " \\\\\n".join(" & ".join(cells[i:i+4]) for i in range(0, len(cells), 4)) + " \\\\"
    # ---------------- step 4: calibration / enrichment / stability / baselines ----------------
    ST4 = pd.read_csv(f"{s4}/stability_by_seed.csv")
    CAL4 = pd.read_csv(f"{s4}/calibration_scores.csv")
    EN4 = pd.read_csv(f"{s4}/enrichment_by_seed.csv")
    REL4 = pd.read_csv(f"{s4}/reliability_table_seed42.csv")
    ENCI4 = pd.read_csv(f"{s4}/enrichment_ci_seed42.csv").set_index("subset")
    BASE4 = pd.read_csv(f"{s4}/regressor_baselines.csv").set_index("model")
    nseed = ST4.seed.nunique()
    def ms(series, f=f3): return f"{f(series.mean())} $\\pm$ {f(series.std(ddof=1))}"
    CH = "chalcogenides + pnictides"; AL = "all compounds"
    ca = CAL4[CAL4.subset == CH]; cl = CAL4[CAL4.subset == AL]
    en = EN4[(EN4.top_fraction == 0.05)]
    def enr(sub, rk, col): return en[(en.subset == sub) & (en.ranking == rk)][col]
    OURS = "window probability (ours)"; N1 = "naive: single-stage closest to 1.4 eV"; N2 = "naive: two-stage closest to 1.4 eV"
    wins1 = int(sum(enr(CH, OURS, "enrichment").values > enr(CH, N1, "enrichment").values))
    wins2 = int(sum(enr(CH, OURS, "enrichment").values > enr(CH, N2, "enrichment").values))
    base_all = float(cl.base_rate.iloc[0]); base_ch = float(ca.base_rate.iloc[0])
    p_ours = enr(CH, OURS, "precision").mean(); e_ours = enr(CH, OURS, "enrichment").mean()
    e_n1 = enr(CH, N1, "enrichment").mean(); e_n2 = enr(CH, N2, "enrichment").mean()
    r_ch = REL4[REL4.subset == CH].copy(); r_ch["dev"] = r_ch.observed_fraction - r_ch.mean_predicted
    top_bin = r_ch.iloc[-1]
    maxdev = float(r_ch.dev.abs().max())
    ridge_mae = float(BASE4.loc["Ridge", "MAE_nonmetals_eV"])
    ridge_note = ("Ridge regression was numerically unstable (MAE above %.0f eV, probably caused by extreme standardised feature values for compounds unlike the training fold) and is not considered further." % ridge_mae
                  if ridge_mae > 5 else f"Ridge regression gave an MAE of {f3(ridge_mae)}~eV.")
    T_cal = "\n".join(row(("all compounds" if k == AL else "chalcogenides and pnictides"), pc(g.base_rate.iloc[0]), ms(g.brier), ms(g.brier_constant_base_rate),
                           ms(g.brier_skill_score), ms(g.AUC_in_window)) for k, g in [(AL, cl), (CH, ca)])
    T_rel = "\n".join(row(int(r.bin) + 1, int(r.n), f3(r.mean_predicted), f3(r.observed_fraction)) for r in r_ch.itertuples())
    rk_names = {OURS: "Window probability", N1: "Naive 1.4~eV, single-stage", N2: "Naive 1.4~eV, two-stage"}
    T_enr = "\n".join(row("all compounds" if sub == AL else "chalcogenides and pnictides", rk_names[rk],
                           ms(enr(sub, rk, "precision")), ms(enr(sub, rk, "enrichment"), f2)) for sub in (AL, CH) for rk in (OURS, N1, N2))
    T_base = "\n".join([row("Random forest (used)", f"{f3(BASE4.loc['Random forest'].MAE_nonmetals_eV)} [{f3(BASE4.loc['Random forest'].ci_low)}, {f3(BASE4.loc['Random forest'].ci_high)}]"),
                         row("Gradient boosting", f"{f3(BASE4.loc['Gradient boosting'].MAE_nonmetals_eV)} [{f3(BASE4.loc['Gradient boosting'].ci_low)}, {f3(BASE4.loc['Gradient boosting'].ci_high)}]"),
                         row("Ridge regression", "unstable (see text)" if ridge_mae > 5 else f3(ridge_mae))])
    cal_sent = (f"Averaged over {nseed} cross-validation seeds, the Brier skill score of the window probability relative to a constant base-rate forecast was "
                f"{ms(ca.brier_skill_score)} for chalcogenides and pnictides and {ms(cl.brier_skill_score)} for all compounds, and the ROC AUC for identifying "
                f"in-window compounds was {ms(ca.AUC_in_window)} and {ms(cl.AUC_in_window)}, respectively (Table~\\ref{{tab:cal}}). "
                f"In the reliability table for chalcogenides and pnictides (Table~\\ref{{tab:rel}}, Fig.~\\ref{{fig:rel}}), predicted and observed frequencies differed by at most {f2(maxdev)} in any bin; "
                f"the highest-probability bin had a mean predicted probability of {f2(top_bin.mean_predicted)} against an observed frequency of {f2(top_bin.observed_fraction)}, "
                f"so the probabilities were reasonably calibrated but not exact.")
    enr_sent = (f"Among chalcogenides and pnictides, {pc(base_ch)} of compounds have a measured gap in the window. Selecting the 5\\% of compounds with the highest window probability gave a precision of "
                f"{f2(p_ours)}, an enrichment of {f2(e_ours)} over random selection (seed-42 bootstrap 95\\% CI {f2(ENCI4.loc[CH].enrichment_ci95_low)} to {f2(ENCI4.loc[CH].enrichment_ci95_high)}). "
                f"The naive ranking by closeness of the single-stage prediction to 1.4~eV gave {f2(e_n1)} and the same ranking with the two-stage model gave {f2(e_n2)}. "
                f"The window-probability ranking had the higher enrichment in {wins1} of {nseed} seeds against the first naive ranking and in {wins2} of {nseed} against the second, "
                f"so its advantage over the single-stage ranking is clearer than over the two-stage ranking. For all compounds the enrichment is higher ({f2(enr(AL, OURS, 'enrichment').mean())}) "
                f"mainly because only {pc(base_all)} of compounds lie in the window, most of the rest being metals that the first stage removes; the value for chalcogenides and pnictides is the relevant one for screening.")
    stab_sent = (f"Across {nseed} different assignments of chemical systems to folds the MAE over all compounds was {ms(ST4.MAE_single)}~eV (single-stage) and {ms(ST4.MAE_two_stage)}~eV (two-stage), "
                 f"and for true non-metals {ms(ST4.MAE_nonmetal_single)} and {ms(ST4.MAE_nonmetal_two_stage)}~eV; metal/non-metal accuracy was {ms(ST4.accuracy)}. The conclusions of Section~\\ref{{sec:single}} were therefore not sensitive to the fold assignment. "
                 f"For the gap regressor on non-metals, gradient boosting (MAE {f3(BASE4.loc['Gradient boosting'].MAE_nonmetals_eV)}~eV) and random forest ({f3(BASE4.loc['Random forest'].MAE_nonmetals_eV)}~eV) had overlapping 95\\% intervals (Table~\\ref{{tab:base}}). {ridge_note}")
    # ---------------- derived sentences ----------------
    gain_nm_sig = (tn_lo <= sn_hi) and (sn_lo <= tn_hi)    # CIs overlap
    nm_sent = (f"For compounds that are truly non-metals, however, the two-stage MAE ({f3(tn)}~eV) and the single-stage MAE ({f3(sn)}~eV) "
               f"had overlapping 95\\% intervals, so we find no clear gain in gap accuracy for non-metals; the improvement comes from "
               f"avoiding the blending of metals and semiconductors." if gain_nm_sig else
               f"For true non-metals the two-stage MAE ({f3(tn)}~eV) differed from the single-stage MAE ({f3(sn)}~eV) beyond overlapping intervals.")
    if NL >= 15:
        lit_abs = (f"Against {NL} independent literature materials the MAE was {f2(LV['MAE_eV'])}~eV.")
        lit_sec = ("Table~\\ref{tab:lit} compares predictions with measured gaps for "
                   f"{NL} materials absent from the training data. The MAE was {f3(LV['MAE_eV'])}~eV (bias, predicted minus measured, "
                   f"{f3(LV['bias_pred_minus_measured_eV'])}~eV); {pc(LV['within_0p5_eV'])} of predictions were within 0.5~eV of the measured value.")
        lit_caveat = ""
    else:
        lit_abs = (f"A preliminary check on only {NL} literature materials is reported but is too small for statistical conclusions.")
        lit_sec = ("Table~\\ref{tab:lit} compares predictions with measured literature gaps for "
                   f"{NL} materials that are absent from the training data. All {len(found)} appear among the top {maxrank} of the {cand_n:,} ranked candidates "
                   f"(they were chosen from the top of the ranking, not at random), all are chalcogenides, and {sys_in} of {len(found)} have a chemical system that also occurs in the training data. "
                   f"This is therefore a consistency check at the top of the ranking, not a test of generalisation. The MAE was {f3(LV.get('MAE_eV', float('nan')))}~eV, all {NL} predictions were within 0.5~eV of the "
                   f"measured value, and all {NL} were assigned a probability of at least 0.30 of lying in the window. With $n={NL}$ no statistical conclusion can be drawn.")
    pdict = dict(
        TITLE="Two-Stage Machine-Learning Screening of Earth-Abundant Semiconductor Absorbers from Composition: Accuracy, Limits Across Anion Families, and Window Probabilities",
        AUTHOR=tx(AUTHOR), AFFIL=tx(AFFIL), EMAIL=tx(EMAIL),
        SUP=(f"\\\\ \\small Supervisor: {tx(SUPERVISOR)}" if SUPERVISOR else ""),
        NDS=nds, NSYS=nsys, NMSHARE=pc(nmshare),
        SA=f3(sa), TA=f3(ta), DD=f3(dd), DD_LO=f3(dd_lo), DD_HI=f3(dd_hi), SN=f3(sn), TN=f3(tn), RN=f3(rn),
        AC=pc(ac), AUC=f3(auc), Q90=f2(q90),
        CH=f3(chal.regressor_MAE), PN=f3(pnic.regressor_MAE), OX=f3(oxid.regressor_MAE), HA=f3(hal.regressor_MAE),
        OXTWO=f3(oxid.two_stage_MAE_nonmetals), OXACC=pc(oxid.accuracy),
        FH_CH=f3(FH.loc["chalcogenide"].MAE_regressor_nonmetals), FH_PN=f3(FH.loc["pnictide"].MAE_regressor_nonmetals),
        FH_OX=f3(FH.loc["oxide"].MAE_regressor_nonmetals), FH_HA=f3(FH.loc["halide"].MAE_regressor_nonmetals),
        PF_NM_S=f3(pf.MAE_single_nonmetals), PF_NM_T=f3(pf.MAE_two_stage_nonmetals), PF_ACC=f3(pf.metal_vs_nonmetal_accuracy),
        PF_AUC=f3(pf.metal_vs_nonmetal_AUC), PS_ACC=f3(ps.metal_vs_nonmetal_accuracy),
        PR_SA=f3(pr.MAE_single_all), PS_SA=f3(ps.MAE_single_all),
        NCAND=f"{cand_n:,}", NCAND30=cand30, PMAX=f2(pmax), NLIT=NL,
        OX_LINE=ox_line, NM_SENT=nm_sent, CAL_SENT=cal_sent, ENR_SENT=enr_sent, STAB_SENT=stab_sent,
        T_CAL=T_cal, T_REL=T_rel, T_ENR=T_enr, T_BASE=T_base, NSEED=nseed,
        BSS_CH=f2(ca.brier_skill_score.mean()), AUC_CH=f2(ca.AUC_in_window.mean()), PREC_CH=pc(p_ours), BASE_CH=pc(base_ch), ENR_CH=f2(e_ours), LIT_ABS=lit_abs, LIT_SEC=lit_sec,
        T_MAIN=T_main, T_FAM=T_fam, T_PROT=T_prot, T_FH=T_fh, T_CF=T_cf, T_ABUND=T_abund, T_TOP=T_top, T_LIT=T_lit,
        CODE_URL=tx(CODE_URL), ALLOWED=", ".join(sorted(e for e in
            ["Li","Na","K","Mg","Ca","Sr","Ba","Al","Si","Ti","V","Cr","Mn","Fe","Ni","Cu","Zn","Sn","Zr","O","S","N","P","Cl","F","Br"])))
    return pdict, NL

TEMPLATE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{booktabs,graphicx,amsmath,hyperref,float,caption,setspace}
\hypersetup{colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue}
\setstretch{1.12}
% DRAFT NOTICE: verify every literature row (value, method, DOI), every reference, and the element-abundance values before submission.
\title{@@TITLE@@}
\author{@@AUTHOR@@\\ \small @@AFFIL@@ \\ \small \texttt{@@EMAIL@@}@@SUP@@}
\date{}
\begin{document}\sloppy\maketitle

\begin{abstract}
Low-cost electronics and photovoltaics need semiconductors made of abundant, non-toxic elements, but experimental screening is slow.
We trained random-forest models on @@NDS@@ experimental band gaps (Matbench \texttt{expt\_gap}) and compared a single-stage regressor with a two-stage model
that first classifies a compound as metal or non-metal and then predicts the gap of non-metals. Under cross-validation that withholds whole chemical systems,
the two-stage model lowered the mean absolute error (MAE) over all compounds from @@SA@@ to @@TA@@~eV (paired difference @@DD@@~eV, 95\% CI @@DD_LO@@ to @@DD_HI@@).
The gain came from the metal/non-metal decision (accuracy @@AC@@); for true non-metals the two models were not clearly different (@@SN@@ vs.\ @@TN@@~eV).
Accuracy depended strongly on anion family: with regressor MAE of @@CH@@~eV for chalcogenides, @@PN@@~eV for pnictides, @@OX@@~eV for oxides and @@HA@@~eV for halides.
When an entire anion family was withheld, the regressor MAE on non-metals was @@FH_PN@@~eV for pnictides and @@FH_CH@@~eV for chalcogenides but @@FH_OX@@~eV for oxides and @@FH_HA@@~eV for halides, and for non-metals the two-stage rule was worse than the single-stage model (@@PF_NM_T@@ vs.\ @@PF_NM_S@@~eV).
Out-of-fold residuals were converted into the probability that a predicted gap lies in the 1.0--1.8~eV single-junction window and applied to @@NCAND@@ Materials Project
compounds made only of abundant, non-toxic elements; @@NCAND30@@ had a probability of at least 0.30 (maximum @@PMAX@@). @@LIT_ABS@@
On held-out compounds the window probability was reasonably calibrated (Brier skill score @@BSS_CH@@ for chalcogenides and pnictides), and selecting the top 5\% by this probability gave a precision of @@PREC_CH@@ against a base rate of @@BASE_CH@@ (enrichment @@ENR_CH@@). The results support composition-only screening as a coarse filter for chalcogenides and pnictides, but not for oxides or for unseen anion families, and they do not show that any candidate will work in a device.
\end{abstract}
\noindent\textbf{Keywords:} band gap; machine learning; earth-abundant semiconductors; photovoltaics; materials screening; model validation

\section{Introduction}
The band gap largely determines whether a material is useful as a light absorber, a transparent conductor or an insulator.
Cheap solar cells and low-cost electronics therefore motivate a search for semiconductors that avoid scarce, costly or toxic elements. First-principles calculations
with standard density functionals systematically underestimate gaps and are expensive, which has led to machine-learning models trained directly on experimental
gaps using composition alone \cite{Zhuo2018,Ward2016}. Standard benchmarks such as Matbench allow such models to be compared on common data \cite{Dunn2020}.

Three practical issues are less often quantified. First, experimental data sets contain many metals with a recorded gap of zero, and a single regressor must blend them with semiconductors.
Second, random cross-validation can overstate performance when related compounds appear on both sides of a split, and performance on new chemistries may be lower \cite{Meredig2018}.
Third, a screening user needs to know how likely a predicted gap is to fall in a useful window, not only a point estimate.

This study addresses these issues on a public data set with free software. We (i) compare single-stage and two-stage models under chemical-system cross-validation with intervals that resample whole chemical systems;
(ii) report accuracy per anion family and under a stress test that withholds an entire family; (iii) convert out-of-fold errors into a probability of lying in the 1.0--1.8~eV window, test its calibration and its enrichment of in-window compounds on held-out data, and apply it to real,
low-cost candidate compounds; and (iv) check predictions against measured gaps from the literature. We do not claim to have discovered a new solar absorber; a suitable gap is necessary but far from sufficient for a working device.

\section{Data and Methods}
\subsection{Experimental band-gap data}
We used the \texttt{matbench\_expt\_gap} data set distributed with \texttt{matminer} \cite{Ward2018matminer,Dunn2020}, derived from the compilation of Zhuo et al.\ \cite{Zhuo2018}:
@@NDS@@ inorganic compositions with experimental gaps in eV, of which @@NMSHARE@@ are non-metals (gap~$>0$) and the rest have a recorded gap of exactly zero. No rows were removed or altered.
The compounds fall into @@NSYS@@ distinct chemical systems (a chemical system is the set of elements in a compound), so most systems contain a single compound. Composition cannot encode polymorph, defects,
temperature, or whether a value is optical or fundamental; this label noise is irreducible here. Compounds were assigned to an anion family by priority: oxide (contains O), chalcogenide (S, Se or Te), halide, pnictide (N, P, As or Sb), otherwise ``other'' (mainly metals and intermetallics).

\subsection{Features and models}
Compositions were converted to the Magpie elemental-property descriptors (132 features) with \texttt{pymatgen} and \texttt{matminer} \cite{Ong2013,Ward2018matminer,Ward2016}.
All models were random forests \cite{Breiman2001} from \texttt{scikit-learn} \cite{Pedregosa2011} with fixed settings (200 trees in cross-validation, 300 in final models; no tuning).
The \emph{single-stage} model regresses the gap on all compounds. The \emph{two-stage} model uses a random-forest classifier for metal vs.\ non-metal and a random-forest regressor trained on non-metals only;
a compound is predicted to have gap zero if the classifier's non-metal probability is 0.5 or lower, otherwise the regressor's value.

\subsection{Evaluation}
Main results use five-fold cross-validation in which every chemical system lies wholly in one fold (assignment by a seeded, size-balanced procedure; disjointness was asserted in code).
For comparison we also report ordinary shuffled five-fold cross-validation with a fixed seed, and a stress test in which one anion family at a time (chalcogenide, halide, oxide, pnictide) is withheld entirely, with the ``other'' family always kept in training.
We report MAE and, for metal/non-metal classification, accuracy and the area under the ROC curve. Because compounds within a system are correlated, 95\% intervals come from a bootstrap that resamples whole chemical systems (2000 resamples).
The paired difference between models uses the same resampled systems.

\subsection{Low-cost candidate set}
An element was considered low-cost if its approximate crustal abundance is at least 2~ppm (rounded handbook-style values, Table~\ref{tab:abund}; to be checked against \cite{CRC}), it is not toxic (Pb, Cd, As, Se, Hg, Tl, Be excluded) and it is not scarce or costly (Te, In, Ga, Ge, Sb, Bi, Ag, Co excluded).
Candidate compounds were taken from the Materials Project \cite{Jain2013} (via the \texttt{matbench\_mp\_e\_form} data set): reduced formulas with negative formation energy (the lowest value across polymorphs),
made only of allowed elements, containing at least one of O, S, N or P, no halogen, at most four elements, and not present in the experimental training set by composition.
Negative formation energy is only a rough stability screen; it does not guarantee stability against decomposition. The allowed elements were @@ALLOWED@@.

\subsection{Probability of lying in the target window}
Single-junction photovoltaic absorbers are most efficient for gaps of roughly 1.0--1.8~eV, centred near 1.3--1.4~eV \cite{SQ1961}; we used 1.0--1.8~eV as the target window.
For a candidate with predicted non-metal probability $p$ and regressor prediction $g$, the probability of lying in the window was estimated as
$p \times \frac{1}{n}\sum_i \mathbf{1}[1.0 \le g + r_i \le 1.8]$, where $r_i$ are the out-of-fold residuals of the regressor on true non-metals under the chemical-system protocol.
This assumes that a candidate's error resembles the out-of-fold errors, which is least likely to hold for chemistries unlike the training data.
The 90th percentile of absolute out-of-fold error was @@Q90@@~eV.

\subsection{Calibration, enrichment, stability and baselines}
To test whether the window probability is useful, each outer training set was split by chemical system into a proper-training set (80\%) and a calibration set (20\%); models were fitted on the proper-training set, calibration residuals defined the probabilities, and only the held-out fold was scored.
A compound is \emph{truly in the window} if its measured gap lies in 1.0--1.8~eV (metals, with gap zero, are not). We report the Brier score against a constant base-rate forecast (Brier skill score), the ROC AUC, a reliability table, and the \emph{enrichment}: the precision among the top 2\%, 5\% and 10\% of compounds ranked by window probability divided by the base rate.
Two naive rankings serve as comparison: closeness of the single-stage (or two-stage) predicted gap to 1.4~eV. The whole procedure was repeated for @@NSEED@@ different assignments of systems to folds (seeds 42--44). We compared random forest with histogram gradient boosting (300 iterations, learning rate 0.05) and ridge regression ($\alpha=1$) for the non-metal gap regressor.

\section{Results}
\subsection{Single-stage versus two-stage under chemical-system folds}\label{sec:single}
Table~\ref{tab:main} gives the main results and Fig.~\ref{fig:parity} shows predictions. The two-stage model lowered the MAE over all compounds from @@SA@@ to @@TA@@~eV
(difference @@DD@@~eV, 95\% CI @@DD_LO@@ to @@DD_HI@@). Metal/non-metal accuracy was @@AC@@ (ROC AUC @@AUC@@). @@NM_SENT@@
Fig.~\ref{fig:parity} also shows the cost of the hard rule: non-metals misclassified as metals appear as predictions of exactly zero.
Ordinary shuffled folds gave a single-stage MAE of @@PR_SA@@~eV against @@PS_SA@@~eV for system folds, a negligible difference, because @@NSYS@@ systems share @@NDS@@ compounds and system grouping removes little related information;
chemical-system folds are therefore not a strict test of transfer to new chemistry (Table~\ref{tab:prot}).

\begin{table}[H]\centering\small\caption{Chemical-system cross-validation (eV; 95\% intervals resample whole chemical systems).}\label{tab:main}
\begin{tabular}{lc}\toprule Quantity & Value [95\% CI]\\\midrule
@@T_MAIN@@
\bottomrule\end{tabular}\end{table}

\begin{figure}[H]\centering\includegraphics[width=.9\linewidth]{fig1_single_vs_two_stage.png}\caption{Out-of-fold predictions under chemical-system folds for the single-stage and two-stage models.}\label{fig:parity}\end{figure}

\subsection{Dependence on anion family}
Table~\ref{tab:fam} reports performance by family under chemical-system folds. The regressor error on non-metals was lowest for chalcogenides (@@CH@@~eV) and pnictides (@@PN@@~eV) and higher for oxides (@@OX@@~eV) and halides (@@HA@@~eV).
For oxides only @@OXACC@@ of compounds were correctly classified as metal or non-metal, and the two-stage MAE on oxide non-metals was @@OXTWO@@~eV, much larger than the regressor error given the correct class.

\begin{table}[H]\centering\small\caption{Per-family results under chemical-system folds. MAE (eV) of the regressor on true non-metals with 95\% CI; last column is the two-stage MAE on true non-metals.}\label{tab:fam}
\begin{tabular}{lccccc}\toprule Family & $n$ & Non-metal share & Class.\ accuracy & Regressor MAE [95\% CI] & Two-stage MAE\\\midrule
@@T_FAM@@
\bottomrule\end{tabular}\end{table}

\subsection{Protocol comparison and withholding an entire anion family}
Table~\ref{tab:prot} compares three protocols. Withholding a whole anion family is much harder: classification accuracy fell to @@PF_ACC@@ (AUC @@PF_AUC@@), and the two-stage MAE on non-metals (@@PF_NM_T@@~eV) was larger than the single-stage MAE (@@PF_NM_S@@~eV).
Table~\ref{tab:fh} gives the family-by-family result: when chalcogenides or pnictides were withheld the regressor MAE on non-metals was @@FH_CH@@ and @@FH_PN@@~eV, but @@FH_OX@@~eV for oxides and @@FH_HA@@~eV for halides.
Predictions should therefore not be trusted for anion families absent from the training data.

\begin{table}[H]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}\caption{Evaluation protocols (eV). ``Family'' withholds one anion family at a time; ``other'' is always kept in training and not scored. NM = true non-metals; Reg.\ = regressor given a non-metal.}\label{tab:prot}
\begin{tabular}{lcccccccc}\toprule Protocol & $n$ tested & MAE single & MAE two-stage & NM single & NM two-stage & NM Reg. & Accuracy & AUC\\\midrule
@@T_PROT@@
\bottomrule\end{tabular}\end{table}

\begin{table}[H]\centering\small\caption{Withholding one anion family at a time.}\label{tab:fh}
\begin{tabular}{lcccc}\toprule Held-out family & $n$ & Non-metals & Regressor MAE (eV) & Class.\ accuracy\\\midrule
@@T_FH@@
\bottomrule\end{tabular}\end{table}

\subsection{Calibration and enrichment of the window probability}
@@CAL_SENT@@

\begin{table}[H]\centering\small\caption{Skill of the window probability on held-out compounds (mean $\pm$ s.d.\ over @@NSEED@@ seeds). Brier scores: lower is better; AUC for identifying in-window compounds.}\label{tab:cal}
\begin{tabular}{lccccc}\toprule Subset & Base rate & Brier & Brier (constant) & Brier skill score & AUC\\\midrule
@@T_CAL@@
\bottomrule\end{tabular}\end{table}

\begin{table}[H]\centering\small\caption{Reliability table for chalcogenides and pnictides (seed 42): compounds sorted by predicted window probability into equal-count bins.}\label{tab:rel}
\begin{tabular}{cccc}\toprule Bin & $n$ & Mean predicted & Observed fraction\\\midrule
@@T_REL@@
\bottomrule\end{tabular}\end{table}

\begin{figure}[H]\centering\includegraphics[width=.95\linewidth]{fig_reliability.png}\caption{Reliability of the window probability (seed 42). Dashed line: perfect calibration.}\label{fig:rel}\end{figure}

@@ENR_SENT@@

\begin{table}[H]\centering\footnotesize\caption{Precision and enrichment of the top 5\% of compounds by each ranking (naive rankings sort by closeness of the predicted gap to 1.4~eV) (mean $\pm$ s.d.\ over @@NSEED@@ seeds).}\label{tab:enr}
\begin{tabular}{llcc}\toprule Subset & Ranking & Precision & Enrichment\\\midrule
@@T_ENR@@
\bottomrule\end{tabular}\end{table}

\begin{figure}[H]\centering\includegraphics[width=.65\linewidth]{fig_enrichment.png}\caption{Enrichment over random selection for chalcogenides and pnictides (seed 42).}\label{fig:enr}\end{figure}

\subsection{Stability across splits and baseline regressors}
@@STAB_SENT@@

\begin{table}[H]\centering\small\caption{Non-metal gap regressors under chemical-system folds (seed 42): MAE on true non-metals (eV) with 95\% CI.}\label{tab:base}
\begin{tabular}{lc}\toprule Regressor & MAE [95\% CI]\\\midrule
@@T_BASE@@
\bottomrule\end{tabular}\end{table}

\subsection{Low-cost candidates}
The candidate set contained @@NCAND@@ compounds. Table~\ref{tab:cf} gives counts by family and Table~\ref{tab:top} lists the twelve chalcogenide and pnictide candidates with the highest estimated probability of lying in the 1.0--1.8~eV window.
The highest probability was @@PMAX@@, and @@NCAND30@@ candidates exceeded 0.30 (Fig.~\ref{fig:hist}). Even the best candidates carry a large chance of lying outside the window. Candidates that are oxides are not tabulated:
the per-family results above show that oxide predictions are unreliable, and @@OX_LINE@@.

\begin{table}[H]\centering\small\caption{Candidate counts by family.}\label{tab:cf}
\begin{tabular}{lccc}\toprule Family & Candidates & $P\ge 0.30$ & Maximum $P$\\\midrule
@@T_CF@@
\bottomrule\end{tabular}\end{table}

\begin{table}[H]\centering\small\caption{Highest-ranked chalcogenide and pnictide candidates: predicted gap (eV), non-metal probability, probability of lying in 1.0--1.8~eV, and formation energy (eV/atom). $^\dagger$ Also in Table~\ref{tab:lit}.}\label{tab:top}
\begin{tabular}{lccccc}\toprule Formula & Family & Pred.\ gap & $P$(non-metal) & $P$(window) & $E_\mathrm{form}$\\\midrule
@@T_TOP@@
\bottomrule\end{tabular}\end{table}

\begin{figure}[H]\centering\includegraphics[width=.6\linewidth]{fig2_probability_hist.png}\caption{Distribution of the estimated probability that a candidate's gap lies in 1.0--1.8~eV.}\label{fig:hist}\end{figure}

\subsection{Comparison with measured literature gaps}
@@LIT_SEC@@

\begin{table}[H]\centering\small\caption{Predictions against measured literature gaps for materials absent from the training data (eV). Where a source reports several values the mean is shown. Sources: \cite{Reddy2006,Marquez2016,Hou2017,Budniak2021}.}\label{tab:lit}
\begin{tabular}{lccccc}\toprule Formula & Measured & Predicted & $P$(non-metal) & $P$(window) & $|$Error$|$\\\midrule
@@T_LIT@@
\bottomrule\end{tabular}\end{table}
\begin{figure}[H]\centering\includegraphics[width=.5\linewidth]{fig_literature_validation.png}\caption{Predicted versus measured literature gaps; the shaded band is the 1.0--1.8~eV window.}\label{fig:lit}\end{figure}

\section{Discussion}
Separating metals from non-metals is the main benefit of the two-stage model; it does not make the gap predictions for non-metals clearly better, and its hard rule creates large errors when a non-metal is called a metal, which was most damaging for oxides.
Accuracy varies strongly with chemistry: errors of about @@CH@@~eV for chalcogenides are small relative to the 0.8~eV width of the target window, whereas errors of @@OX@@~eV for oxides are not. Because the overall 90th-percentile error of @@Q90@@~eV exceeds that width,
the model cannot confirm that a compound's gap is suitable. It can, however, enrich a list: on held-out chalcogenides and pnictides the top 5\% by window probability contained in-window compounds @@ENR_CH@@ times as often as random selection, which supports its use as a first filter. The stress test shows that these statements hold only for anion families represented in the training data.

Practically, composition-only screening of this kind is a coarse, inexpensive first filter for chalcogenides and pnictides built from abundant elements. It ignores crystal structure, defects, doping, carrier lifetime, stability and processing, all of which determine whether a material works in a device.
Candidates are hypotheses for further calculation or experiment, not demonstrated absorbers.

\paragraph{Limitations.}
(i) Composition alone cannot distinguish polymorphs, defects or measurement conditions, and the compiled data contain label noise. (ii) Chemical-system grouping is a weak test because most systems contain one compound; the family-withholding test is much stricter but also unrealistically severe for most uses.
(iii) The window probability assumes that candidate errors resemble out-of-fold errors. (iv) Negative formation energy is not a proof of stability. (v) Element abundance values are rounded handbook figures and the toxicity and cost lists are judgements.
(vi) The literature comparison is @@NLIT@@ materials taken from the top of the ranking; it does not test generalisation and has no statistical power. (vii) Models used fixed settings without tuning, so absolute accuracy is not state of the art. (viii) Calibration and enrichment were measured on compounds from the same data set, with @@NSEED@@ fold assignments, so they describe performance on compounds like those in Matbench, not on genuinely new chemistries; the advantage of the window-probability ranking over the two-stage naive ranking is small.

\section{Conclusions}
Under chemical-system cross-validation on @@NDS@@ experimental compositions, a two-stage model lowered the overall MAE from @@SA@@ to @@TA@@~eV by handling metals separately, without a clear gain for non-metal gaps.
Accuracy was acceptable for chalcogenides and pnictides, poor for oxides and halides, and degraded sharply when an anion family was unseen. A probability-of-window ranking, which was reasonably calibrated and enriched in-window chalcogenides and pnictides @@ENR_CH@@-fold at the top 5\% on held-out data, applied to @@NCAND@@ real low-cost compounds yielded @@NCAND30@@ candidates above 0.30,
none of which has been shown here to be a working absorber. A larger independent literature validation is the most important next step.

\section*{Data and Code Availability}
Code, fold assignments, predictions and result tables are available at \url{@@CODE_URL@@}. The experimental data are the public Matbench \texttt{matbench\_expt\_gap} set; candidate compositions come from the public Materials Project data distributed with \texttt{matminer}.

\section*{Declaration of AI Assistance}
Generative AI (Anthropic's Claude) was used to help write the analysis code and draft the manuscript. The author ran the code, checked the results and takes responsibility for the content. \emph{(Edit to match your institution's and the journal's policy.)}

\section*{Funding and Competing Interests}
No external funding. No competing interests declared. \emph{(Confirm before submission.)}

\appendix
\section{Element abundance values used}
\begin{table}[H]\centering\small\caption{Approximate crustal abundance (mg/kg) of the allowed elements (rounded values; verify against \cite{CRC} before submission).}\label{tab:abund}
\begin{tabular}{lrlrlrlr}\toprule @@T_ABUND@@
\bottomrule\end{tabular}\end{table}

\begin{thebibliography}{99}\small
\bibitem{Dunn2020} A.~Dunn, Q.~Wang, A.~Ganose, D.~Dopp, A.~Jain, Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm, \emph{npj Comput.\ Mater.} \textbf{6}, 138 (2020).
\bibitem{Ward2016} L.~Ward, A.~Agrawal, A.~Choudhary, C.~Wolverton, A general-purpose machine learning framework for predicting properties of inorganic materials, \emph{npj Comput.\ Mater.} \textbf{2}, 16028 (2016).
\bibitem{Zhuo2018} Y.~Zhuo, A.~Mansouri Tehrani, J.~Brgoch, Predicting the band gaps of inorganic solids by machine learning, \emph{J.\ Phys.\ Chem.\ Lett.} \textbf{9}, 1668 (2018).
\bibitem{Meredig2018} B.~Meredig et al., Can machine learning identify the next high-temperature superconductor? Examining extrapolation performance for materials discovery, \emph{Mol.\ Syst.\ Des.\ Eng.} \textbf{3}, 819 (2018).
\bibitem{Ong2013} S.~P.~Ong et al., Python Materials Genomics (pymatgen), \emph{Comput.\ Mater.\ Sci.} \textbf{68}, 314 (2013).
\bibitem{Ward2018matminer} L.~Ward et al., Matminer: an open source toolkit for materials data mining, \emph{Comput.\ Mater.\ Sci.} \textbf{152}, 60 (2018).
\bibitem{Pedregosa2011} F.~Pedregosa et al., Scikit-learn: machine learning in Python, \emph{J.\ Mach.\ Learn.\ Res.} \textbf{12}, 2825 (2011).
\bibitem{Breiman2001} L.~Breiman, Random forests, \emph{Mach.\ Learn.} \textbf{45}, 5 (2001).
\bibitem{Jain2013} A.~Jain et al., Commentary: The Materials Project: a materials genome approach to accelerating materials innovation, \emph{APL Mater.} \textbf{1}, 011002 (2013).
\bibitem{SQ1961} W.~Shockley, H.~J.~Queisser, Detailed balance limit of efficiency of p--n junction solar cells, \emph{J.\ Appl.\ Phys.} \textbf{32}, 510 (1961).
\bibitem{CRC} W.~M.~Haynes (ed.), \emph{CRC Handbook of Chemistry and Physics}, CRC Press (current edition; abundance of the elements in the Earth's crust).
\bibitem{Reddy2006} N.~Koteeswara Reddy et al., Synthesis and characterisation of co-evaporated tin sulphide thin films, \emph{Appl.\ Phys.\ A} \textbf{83}, 133 (2006). doi:10.1007/s00339-005-3475-y
\bibitem{Marquez2016} J.~Marquez Prieto, S.~Levcenko, J.~Just, H.~Hampel, I.~Forbes, N.~Pearsall, T.~Unold, Earth abundant thin film solar cells from co-evaporated Cu$_2$SnS$_3$ absorber layers, \emph{J.\ Alloys Compd.} \textbf{689}, 182 (2016). doi:10.1016/j.jallcom.2016.07.293
\bibitem{Hou2017} H.~Hou, H.~Guan, L.~Li, Synthesis of Cu$_2$FeSnS$_4$ thin films with stannite and wurtzite structure directly on glass substrates via the solvothermal method, \emph{J.\ Mater.\ Sci.: Mater.\ Electron.} \textbf{28}, 7745 (2017). doi:10.1007/s10854-017-6469-6
\bibitem{Budniak2021} A.~K.~Budniak et al., Spectroscopy and structural investigation of iron phosphorus trisulfide -- FePS$_3$, arXiv:2108.01742 (2021) [journal reference to be verified].
\end{thebibliography}
\end{document}
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step2", default="step2_results"); ap.add_argument("--step3", default="step3_results")
    ap.add_argument("--step4", default="step4_results"); ap.add_argument("--out", default="paper.tex")
    a = ap.parse_args()
    d, nl = build(a.step2, a.step3, a.step4)
    tex = TEMPLATE
    for k, v in d.items(): tex = tex.replace(f"@@{k}@@", str(v))
    left = re.findall(r"@@[A-Z_0-9]+@@", tex)
    assert not left, f"unfilled placeholders: {left}"
    open(a.out, "w").write(tex)
    here = os.path.dirname(os.path.abspath(a.out)) or "."
    hl = [f"Metal/non-metal step lowers band-gap MAE from {d['SA']} to {d['TA']} eV",
          "Accuracy is good for chalcogenides, poor for oxides and unseen anion families",
          "Window probability is reasonably calibrated on held-out compounds",
          f"Top 5% by probability is {d['ENR_CH']}x richer in 1.0-1.8 eV gaps than random",
          f"Screens {d['NCAND']} abundant, non-toxic Materials Project compounds"]
    assert all(len(h) <= 85 for h in hl), [len(h) for h in hl]
    open(os.path.join(here, "highlights.txt"), "w").write("\n".join(hl) + "\n")
    cover = f"""Dear Editor,

Please consider the manuscript "{d['TITLE']}" for publication in [JOURNAL NAME].

The study asks a narrow question that matters for low-cost semiconductor screening: how far can band gaps predicted from composition alone be trusted, and can the predictions be turned into a calibrated probability of landing in the 1.0-1.8 eV window used for single-junction absorbers? Using {d['NDS']} experimental band gaps, we compare a single-stage and a two-stage (metal/non-metal, then gap) random-forest model under cross-validation that withholds whole chemical systems and whole anion families, quantify performance per family, and test the calibration and enrichment of the window probability on held-out compounds (precision {d['PREC_CH']} in the top 5% against a base rate of {d['BASE_CH']} for chalcogenides and pnictides). The model is then applied to {d['NCAND']} abundant, non-toxic compounds from the Materials Project.

We state the limits plainly: the model is not reliable for oxides or for anion families absent from the training data, a suitable band gap does not imply a working device, and the literature comparison is [N = {d['NLIT']}; UPDATE IF EXPANDED]. All code, fold assignments and result tables are openly available at {CODE_URL}.

The work is original, is not under consideration elsewhere, and all authors approve the submission. Generative AI (Anthropic's Claude) assisted with code and drafting; the author(s) ran, checked and take responsibility for all content. The author(s) declare no competing interests. [CONFIRM ALL STATEMENTS]

Suggested reviewers (optional, no conflicts of interest): [NAME, AFFILIATION, EMAIL] x 3

Sincerely,
{AUTHOR}
{AFFIL}
{EMAIL}
"""
    open(os.path.join(here, "cover_letter_template.txt"), "w").write(cover)
    for src, names in [(a.step2, ["fig1_single_vs_two_stage.png", "fig2_probability_hist.png"]), (a.step3, ["fig_literature_validation.png"]), (a.step4, ["fig_reliability.png", "fig_enrichment.png"])]:
        for n in names:
            if os.path.exists(f"{src}/{n}"): shutil.copy(f"{src}/{n}", here)
    print("wrote", a.out, "| literature materials:", nl)
