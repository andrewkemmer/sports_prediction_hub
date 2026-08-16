// Static MLB team metadata keyed by Stats API team id.
// Used to add abbreviations / short names without the verbose `team` hydrate.

export interface TeamMeta {
  abbrev: string;
  name: string; // location / short name
  fullName: string;
  color: string; // primary brand color (hex)
}

export const TEAMS: Record<number, TeamMeta> = {
  108: { abbrev: "LAA", name: "Los Angeles", fullName: "Los Angeles Angels", color: "#BA0021" },
  109: { abbrev: "ARI", name: "Arizona", fullName: "Arizona Diamondbacks", color: "#A71930" },
  110: { abbrev: "BAL", name: "Baltimore", fullName: "Baltimore Orioles", color: "#DF4601" },
  111: { abbrev: "BOS", name: "Boston", fullName: "Boston Red Sox", color: "#BD3039" },
  112: { abbrev: "CHC", name: "Chicago", fullName: "Chicago Cubs", color: "#4f8ae0" },
  113: { abbrev: "CIN", name: "Cincinnati", fullName: "Cincinnati Reds", color: "#C6011F" },
  114: { abbrev: "CLE", name: "Cleveland", fullName: "Cleveland Guardians", color: "#E31937" },
  115: { abbrev: "COL", name: "Colorado", fullName: "Colorado Rockies", color: "#6a4c93" },
  116: { abbrev: "DET", name: "Detroit", fullName: "Detroit Tigers", color: "#0C2340" },
  117: { abbrev: "HOU", name: "Houston", fullName: "Houston Astros", color: "#EB6E1F" },
  118: { abbrev: "KC", name: "Kansas City", fullName: "Kansas City Royals", color: "#004687" },
  119: { abbrev: "LAD", name: "Los Angeles", fullName: "Los Angeles Dodgers", color: "#005A9C" },
  120: { abbrev: "WSH", name: "Washington", fullName: "Washington Nationals", color: "#AB0003" },
  121: { abbrev: "NYM", name: "New York", fullName: "New York Mets", color: "#002D72" },
  133: { abbrev: "ATH", name: "Athletics", fullName: "Athletics", color: "#2f7d4a" },
  134: { abbrev: "PIT", name: "Pittsburgh", fullName: "Pittsburgh Pirates", color: "#FDB827" },
  135: { abbrev: "SD", name: "San Diego", fullName: "San Diego Padres", color: "#8b5a2b" },
  136: { abbrev: "SEA", name: "Seattle", fullName: "Seattle Mariners", color: "#005C5C" },
  137: { abbrev: "SF", name: "San Francisco", fullName: "San Francisco Giants", color: "#FD5A1E" },
  138: { abbrev: "STL", name: "St. Louis", fullName: "St. Louis Cardinals", color: "#C41E3A" },
  139: { abbrev: "TB", name: "Tampa Bay", fullName: "Tampa Bay Rays", color: "#092C5C" },
  140: { abbrev: "TEX", name: "Texas", fullName: "Texas Rangers", color: "#003278" },
  141: { abbrev: "TOR", name: "Toronto", fullName: "Toronto Blue Jays", color: "#134A8E" },
  142: { abbrev: "MIN", name: "Minnesota", fullName: "Minnesota Twins", color: "#002B5C" },
  143: { abbrev: "PHI", name: "Philadelphia", fullName: "Philadelphia Phillies", color: "#E81828" },
  144: { abbrev: "ATL", name: "Atlanta", fullName: "Atlanta Braves", color: "#CE1141" },
  145: { abbrev: "CWS", name: "Chicago", fullName: "Chicago White Sox", color: "#27251F" },
  146: { abbrev: "MIA", name: "Miami", fullName: "Miami Marlins", color: "#00A3E0" },
  147: { abbrev: "NYY", name: "New York", fullName: "New York Yankees", color: "#4c6b9e" },
  158: { abbrev: "MIL", name: "Milwaukee", fullName: "Milwaukee Brewers", color: "#0A2351" },
};

export function teamMeta(id: number): TeamMeta {
  return TEAMS[id] ?? { abbrev: "TBD", name: `Team ${id}`, fullName: `Team ${id}`, color: "#8b93a7" };
}

// Approximate multi-year ballpark run factors keyed by home team id
// (league average = 1.00). Used as a ballpark-context feature and a prior
// for the run-scoring model. Coors Field is the most hitter-friendly park.
export const PARK_FACTORS: Record<number, number> = {
  108: 0.98, // LAA — Angel Stadium
  109: 1.02, // ARI — Chase Field
  110: 1.03, // BAL — Oriole Park at Camden Yards
  111: 1.04, // BOS — Fenway Park
  112: 1.02, // CHC — Wrigley Field
  113: 1.07, // CIN — Great American Ball Park
  114: 1.0, // CLE — Progressive Field
  115: 1.19, // COL — Coors Field
  116: 0.96, // DET — Comerica Park
  117: 0.99, // HOU — Minute Maid Park
  118: 1.05, // KC — Kauffman Stadium
  119: 0.98, // LAD — Dodger Stadium
  120: 0.99, // WSH — Nationals Park
  121: 0.94, // NYM — Citi Field
  133: 0.97, // ATH — Sutter Health Park
  134: 0.97, // PIT — PNC Park
  135: 0.95, // SD — Petco Park
  136: 0.95, // SEA — T-Mobile Park
  137: 0.98, // SF — Oracle Park
  138: 0.96, // STL — Busch Stadium
  139: 0.94, // TB — Tropicana Field
  140: 1.03, // TEX — Globe Life Field
  141: 1.01, // TOR — Rogers Centre
  142: 0.99, // MIN — Target Field
  143: 1.02, // PHI — Citizens Bank Park
  144: 1.01, // ATL — Truist Park
  145: 1.0, // CWS — Rate Field
  146: 0.95, // MIA — loanDepot park
  147: 1.02, // NYY — Yankee Stadium
  158: 0.98, // MIL — American Family Field
};
