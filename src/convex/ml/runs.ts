// Pure-TS run-scoring model for MLB game totals and run lines.
// No node/convex imports — safe to use from Convex actions and shared types.
//
// Approach:
//   1. Fit each team's offensive / defensive run factor from historical scores
//      plus a per-ballpark factor (all derived from the single game dataset).
//   2. Model each team's runs as independent Poisson random variables with
//      mean = leagueAvg * offense(home) * defense(away) * park(home).
//   3. Monte Carlo (10,000 simulations) produces predicted scores, the total
//      over/under distribution, and run-line (±1.5) cover probabilities.

import { RawGame } from "./types";

export interface RunModel {
  leagueRuns: number;
  teamOffense: Record<number, number>;
  teamDefense: Record<number, number>;
  parkFactor: Record<number, number>; // keyed by home team id (venue proxy)
}

export interface RunSimulation {
  homeScore: number; // mean runs
  awayScore: number; // mean runs
  total: number; // mean combined runs
  overProb: number; // P(total > line)
  underProb: number; // P(total < line)
  homeRunLineProb: number; // P(home wins by 2+ → covers -1.5)
  awayRunLineProb: number; // P(away loses by ≤1 or wins → covers +1.5)
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

/** Deterministic xorshift PRNG so Monte Carlo results are reproducible. */
function makeRand(seed: number): () => number {
  let s = (seed >>> 0) || 0x12345678;
  return () => {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 4294967296;
  };
}

/** Knuth's Poisson sampler. */
function poisson(lambda: number, rand: () => number): number {
  if (lambda <= 0) return 0;
  const L = Math.exp(-lambda);
  let k = 0;
  let p = 1;
  do {
    k += 1;
    p *= rand();
  } while (p > L);
  return k - 1;
}

export function fitRunModel(games: RawGame[]): RunModel {
  const teamGames = new Map<number, number>();
  const teamScored = new Map<number, number>();
  const teamAllowed = new Map<number, number>();
  const parkRuns = new Map<number, { runs: number; games: number }>();
  let totalRuns = 0;
  let totalGames = 0;

  for (const g of games) {
    if (g.winner !== "home" && g.winner !== "away") continue;
    const hs = g.home.score;
    const as = g.away.score;
    if (typeof hs !== "number" || typeof as !== "number") continue;
    const homeId = g.home.id;
    const awayId = g.away.id;
    teamGames.set(homeId, (teamGames.get(homeId) ?? 0) + 1);
    teamGames.set(awayId, (teamGames.get(awayId) ?? 0) + 1);
    teamScored.set(homeId, (teamScored.get(homeId) ?? 0) + hs);
    teamAllowed.set(homeId, (teamAllowed.get(homeId) ?? 0) + as);
    teamScored.set(awayId, (teamScored.get(awayId) ?? 0) + as);
    teamAllowed.set(awayId, (teamAllowed.get(awayId) ?? 0) + hs);
    const pr = parkRuns.get(homeId) ?? { runs: 0, games: 0 };
    parkRuns.set(homeId, { runs: pr.runs + hs + as, games: pr.games + 1 });
    totalRuns += hs + as;
    totalGames += 1;
  }

  const leagueRuns = totalGames > 0 ? totalRuns / (2 * totalGames) : 4.4;

  const teamOffense: Record<number, number> = {};
  const teamDefense: Record<number, number> = {};
  const parkFactor: Record<number, number> = {};
  for (const id of teamGames.keys()) {
    const g = teamGames.get(id) ?? 1;
    teamOffense[id] = clamp((teamScored.get(id) ?? 0) / g / leagueRuns, 0.7, 1.3);
    teamDefense[id] = clamp((teamAllowed.get(id) ?? 0) / g / leagueRuns, 0.7, 1.3);
  }
  for (const [id, p] of parkRuns) {
    parkFactor[id] = clamp(p.runs / p.games / (2 * leagueRuns), 0.85, 1.15);
  }

  return { leagueRuns, teamOffense, teamDefense, parkFactor };
}

/**
 * Monte Carlo run simulation for a matchup. `line` is the total over/under
 * reference (market total when available, otherwise the model mean total).
 */
export function simulateRuns(
  model: RunModel,
  homeId: number,
  awayId: number,
  line: number,
  trials = 10000,
): RunSimulation {
  const offense = model.teamOffense;
  const defense = model.teamDefense;
  const park = model.parkFactor;
  const parkMul = park[homeId] ?? 1;
  const lambdaHome = model.leagueRuns * (offense[homeId] ?? 1) * (defense[awayId] ?? 1) * parkMul;
  const lambdaAway = model.leagueRuns * (offense[awayId] ?? 1) * (defense[homeId] ?? 1) * parkMul;

  const rand = makeRand((homeId * 1000003 + awayId * 7919) >>> 0);
  let homeSum = 0;
  let awaySum = 0;
  let over = 0;
  let under = 0;
  let homeCover = 0;
  let awayCover = 0;
  for (let t = 0; t < trials; t++) {
    const hs = poisson(lambdaHome, rand);
    const as = poisson(lambdaAway, rand);
    homeSum += hs;
    awaySum += as;
    const total = hs + as;
    if (total > line) over += 1;
    else if (total < line) under += 1;
    const margin = hs - as;
    if (margin >= 2) homeCover += 1;
    if (margin <= 1) awayCover += 1;
  }

  return {
    homeScore: homeSum / trials,
    awayScore: awaySum / trials,
    total: (homeSum + awaySum) / trials,
    overProb: over / trials,
    underProb: under / trials,
    homeRunLineProb: homeCover / trials,
    awayRunLineProb: awayCover / trials,
  };
}
