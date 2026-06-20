#include <utility>
#include <algorithm>
#include "state.hpp"
#include "minimax.hpp"


static bool is_capture_move(State* state, const Move& action){
    int opp = 1 - state->player;
    return state->piece_at(opp, (int)action.second.first, (int)action.second.second) > 0;
}

static int capture_order_score(State* state, const Move& action){
    static const int value[7] = {0, 200, 600, 700, 800, 2000, 20000};
    int self = state->player;
    int opp = 1 - self;
    int victim = state->piece_at(opp, (int)action.second.first, (int)action.second.second);
    int attacker = state->piece_at(self, (int)action.first.first, (int)action.first.second);
    
    // 【MVV-LVA 吃子排序】：優先測試「最有價值受害者 - 最無價值攻擊者」
    // 乘以 16 確保受害者價值具有絕對優先權，同級受害者才比較攻擊者價值。
    return value[victim] * 16 - value[attacker];
}

static int get_qs_limit(State* state, const Move& action){
    int opp = 1 - state->player;
    int victim = state->piece_at(opp, (int)action.second.first, (int)action.second.second);
    
    // 【動態 QS 深度控制】：根據獵物價值動態放寬靜態搜尋的極限深度
    if (victim == 6) return 10; // 吃國王：解鎖深搜至 10 層，精準捕捉絕殺機會！
    if (victim == 5) return 7;  // 吃后：追溯 7 層
    if (victim == 4) return 5;  // 吃車：追溯 5 層
    return 2;                   // 兵/馬/象：最多追溯 2 層
}

static void order_moves(State* state){
    std::sort(state->legal_actions.begin(), state->legal_actions.end(),
        [state](const Move& a, const Move& b){
            return capture_order_score(state, a) > capture_order_score(state, b);
        });
}

static int child_window_score(
    State* parent,
    const Move& action,
    int depth,
    int alpha,
    int beta,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p
){
    State* next = static_cast<State*>(parent->next_state(action));
    bool same = next->same_player_as_parent();
    int raw;
    if(same){
        raw = MiniMax::eval_ctx(next, depth, alpha, beta, history, ply, ctx, p);
    }else{
        raw = MiniMax::eval_ctx(next, depth, -beta, -alpha, history, ply, ctx, p);
    }
    int score = same ? raw : -raw;
    delete next;
    return score;
}

int MiniMax::quiescence_ctx(
    State *state,
    int alpha,
    int beta,
    GameHistory& history,
    int ply,
    int qs_ply,
    SearchContext& ctx,
    const MMParams& p
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    if(state->game_state == WIN){
        return P_MAX - ply;
    }
    if(state->game_state == DRAW){
        return 0;
    }

    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }

    int stand_pat = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
    const int ABSOLUTE_MAX_PLY = 100;
    
    // 【Stand Pat 保底防暴走機制】
    // 若當前靜態評分已大於 beta，代表局面夠好，不吃子也行，直接剪枝退回。
    if(stand_pat >= beta || ply >= ABSOLUTE_MAX_PLY){
        return stand_pat;
    }
    if(stand_pat > alpha){
        alpha = stand_pat;
    }

    history.push(state->hash());
    order_moves(state);

    for(auto& action : state->legal_actions){
        if(!is_capture_move(state, action)){
            continue;
        }
        int limit = get_qs_limit(state, action);
        if (qs_ply >= limit) {
            continue; 
        }

        State* next = static_cast<State*>(state->next_state(action));
        bool same = next->same_player_as_parent();
        int raw;
        if(same){
            raw = quiescence_ctx(next, alpha, beta, history, ply + 1, qs_ply + 1, ctx, p);
        }else{
            raw = quiescence_ctx(next, -beta, -alpha, history, ply + 1, qs_ply + 1, ctx, p);
        }
        int score = same ? raw : -raw;
        delete next;

        if(score >= beta){
            history.pop(state->hash());
            return score;
        }
        if(score > alpha){
            alpha = score;
        }
        if(ctx.stop){
            break;
        }
    }

    history.pop(state->hash());
    return alpha;
}

/*============================================================
 * MiniMax — eval_ctx
 *
 * Negamax without pruning. Caller manages memory.
 *============================================================*/
int MiniMax::eval_ctx(
    State *state,
    int depth,
    int alpha,
    int beta,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    /* === Lazy move generation (sets game_state) === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal / leaf checks === */

    // [ Hackathon TODO 3-1 ]
    // 【快速致勝機制】：減去 ply 確保 AI 在有多種殺法時，會優先選擇「步數最短」的路徑。
    if (state->game_state==WIN) {
        return P_MAX-ply;
    }

    if(state->game_state == DRAW){
        return 0;
    }

    /* === Repetition check (game-specific) === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }
    history.push(state->hash());

    if(depth <= 0){
        int score = p.use_quiescence
            ? quiescence_ctx(state, alpha, beta, history, ply, 0, ctx, p)
            : state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history); 
        history.pop(state->hash());
        return score;
    }

    /* === Negamax loop === */
    order_moves(state);
    int best_score = M_MAX;
    bool first_child = true;

    for(auto& action : state->legal_actions){
        // [ Hackathon TODO 3-2 ]
        int score;
        
        // 【PVS 主要變例搜尋機制】
        if(!p.use_pvs || first_child){
            // ① 首步全視窗 (Full Window)：對排序第一的動作進行完整深搜，確立嚴謹的基準 alpha
            score = child_window_score(state, action, depth - 1, alpha, beta,
                                       history, ply + 1, ctx, p);
        }else{
            // ② 零視窗探測 (Null Window)：針對其餘動作，用極窄的視窗 (alpha, alpha + 1) 探測
            // 由於容錯率低，極易觸發 Beta Cutoff，從而省下海量算力
            score = child_window_score(state, action, depth - 1, alpha, alpha + 1,
                                       history, ply + 1, ctx, p);
                                       
            // ③ 假設錯誤重搜 (Re-search)：若探測發現這步其實更強 (大於 alpha)，則放寬視窗重新深搜
            if(score > alpha && score < beta && !ctx.stop){
                score = child_window_score(state, action, depth - 1, alpha, beta,
                                           history, ply + 1, ctx, p);
            }
        }
        first_child = false;

        // [ Hackathon TODO 3-5 ]
        if(score>best_score){
            best_score=score;
        }
        if(score>alpha){
            alpha=score;
        }
        
        // Beta 剪枝：這步太好了，對手絕對不會讓我走到這個局面，直接 Cutoff！
        if(alpha>=beta || ctx.stop){
            break;
        }

    }

    history.pop(state->hash());
    return best_score;
}


/*============================================================
 * MiniMax — search
 *============================================================*/
SearchResult MiniMax::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();
    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }
    order_moves(state);

    int alpha = M_MAX;
    int beta = P_MAX;
    int best_score = M_MAX;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();
    bool first_child = true;

    for(auto& action : state->legal_actions){
        /* [ Hackathon TODO 4-1 ] */
        int score;
        if(!p.use_pvs || first_child){
            score = child_window_score(state, action, depth - 1, alpha, beta,
                                       history, 1, ctx, p);
        }else{
            score = child_window_score(state, action, depth - 1, alpha, alpha + 1,
                                       history, 1, ctx, p);
            if(score > alpha && score < beta && !ctx.stop){
                score = child_window_score(state, action, depth - 1, alpha, beta,
                                           history, 1, ctx, p);
            }
        }
        first_child = false;

        if(score > best_score){
            best_score = score;
            result.best_move = action;
            if(p.report_partial && ctx.on_root_update){
                ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
            }
        }
        if(score > alpha){
            alpha = score;
        }
        if(ctx.stop){
            break;
        }
        move_index++;
    }

    // [ Hackathon TODO 4-3 ]
    result.score=best_score;
    result.nodes=ctx.nodes;
    result.seldepth=ctx.seldepth;
    if(total_moves > 0){
        result.pv = {result.best_move};
    }
    return result;
} 


/*============================================================
 * MiniMax — default_params / param_defs
 *============================================================*/
ParamMap MiniMax::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"UsePVS", "true"},
        {"UseQuiescence", "true"},
        {"QuiescenceMaxPly", "8"},
        {"ReportPartial", "true"},
    };
}

std::vector<ParamDef> MiniMax::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"UsePVS", ParamDef::CHECK, "true"},
        {"UseQuiescence", ParamDef::CHECK, "true"},
        {"QuiescenceMaxPly", ParamDef::SPIN, "8", 0, 32},
        {"ReportPartial", ParamDef::CHECK, "true"},
    };
}