#pragma once
#include "search_types.hpp"
#include "game_history.hpp"

struct MMParams {
    bool use_kp_eval = true;
    bool use_eval_mobility = true;
    bool use_pvs = true;
    bool use_quiescence = true;
    bool report_partial = true;
    int quiescence_max_ply = 8;

    static MMParams from_map(const ParamMap& m){
        MMParams p;
        p.use_kp_eval       = param_bool(m, "UseKPEval", true);
        p.use_eval_mobility = param_bool(m, "UseEvalMobility", true);
        p.use_pvs           = param_bool(m, "UsePVS", true);
        p.use_quiescence    = param_bool(m, "UseQuiescence", true);
        p.report_partial    = param_bool(m, "ReportPartial", true);
        p.quiescence_max_ply  = param_int(m, "QuiescenceMaxPly", 8);
        return p;
    }
};

class MiniMax{
public:
    static int eval_ctx(
        State *state,
        int depth,
        int alpha,
        int beta,
        GameHistory& history,
        int ply,
        SearchContext& ctx,
        const MMParams& p
    );
    static int quiescence_ctx(
        State *state,
        int alpha,
        int beta,
        GameHistory& history,
        int ply,
        int qs_ply,
        SearchContext& ctx,
        const MMParams& p
    );
    static SearchResult search(
        State *state,
        int depth,
        GameHistory& history,
        SearchContext& ctx
    );

    static ParamMap default_params();
    static std::vector<ParamDef> param_defs();
};
