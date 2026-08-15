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
