/* eslint-disable */
/**
 * Generated `api` utility.
 *
 * THIS CODE IS AUTOMATICALLY GENERATED.
 *
 * To regenerate, run `npx convex dev`.
 * @module
 */

import type * as auth from "../auth.js";
import type * as auth_emailOtp from "../auth/emailOtp.js";
import type * as backfill from "../backfill.js";
import type * as http from "../http.js";
import type * as ml_model from "../ml/model.js";
import type * as ml_runs from "../ml/runs.js";
import type * as ml_teams from "../ml/teams.js";
import type * as ml_types from "../ml/types.js";
import type * as mlb from "../mlb.js";
import type * as mlbActions from "../mlbActions.js";
import type * as users from "../users.js";

import type {
  ApiFromModules,
  FilterApi,
  FunctionReference,
} from "convex/server";

declare const fullApi: ApiFromModules<{
  auth: typeof auth;
  "auth/emailOtp": typeof auth_emailOtp;
  backfill: typeof backfill;
  http: typeof http;
  "ml/model": typeof ml_model;
  "ml/runs": typeof ml_runs;
  "ml/teams": typeof ml_teams;
  "ml/types": typeof ml_types;
  mlb: typeof mlb;
  mlbActions: typeof mlbActions;
  users: typeof users;
}>;

/**
 * A utility for referencing Convex functions in your app's public API.
 *
 * Usage:
 * ```js
 * const myFunctionReference = api.myModule.myFunction;
 * ```
 */
export declare const api: FilterApi<
  typeof fullApi,
  FunctionReference<any, "public">
>;

/**
 * A utility for referencing Convex functions in your app's internal API.
 *
 * Usage:
 * ```js
 * const myFunctionReference = internal.myModule.myFunction;
 * ```
 */
export declare const internal: FilterApi<
  typeof fullApi,
  FunctionReference<any, "internal">
>;

export declare const components: {};
