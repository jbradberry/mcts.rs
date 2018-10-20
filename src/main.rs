#[macro_use]
extern crate serde_derive;

extern crate serde;
extern crate serde_json;

extern crate rand;


use std::cmp;
use std::collections::HashMap;
use std::f64;
use std::io;
use std::io::prelude::*;

use rand::{thread_rng, Rng};


pub trait BoardPlayer {}


pub trait BoardAction {}


pub trait BoardState<A: BoardAction, P: BoardPlayer> {
    fn starting_state() -> Self;

    fn previous_player(&self) -> P;

    fn current_player(&self) -> P;

    fn next_state(&self, action: &A) -> Self;

    fn is_legal(&self, action: &A, history: &[Self]) -> bool where Self: Sized;

    fn legal_actions(&self, history: &[Self]) -> Vec<A> where Self: Sized;

    fn is_ended(&self, history: &[Self]) -> bool where Self: Sized;
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChongPlayer {
    Player1,
    Player2
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChongState {
    pawn1: u64,
    pawn2: u64,
    stones1: u64,
    stones2: u64,
    next: ChongPlayer
}


#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ChongPiece {
    Pawn,
    Stone
}


#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChongAction {
    piece: ChongPiece,
    r: u8,
    c: u8
}


impl BoardPlayer for ChongPlayer {}


impl ChongPlayer {
    fn next_player(&self) -> ChongPlayer {
        match self {
            ChongPlayer::Player1 => ChongPlayer::Player2,
            ChongPlayer::Player2 => ChongPlayer::Player1
        }
    }
}


impl BoardAction for ChongAction {}


impl ChongState {
    fn stones_remaining(&self, player: ChongPlayer) -> u32 {
        match player {
            ChongPlayer::Player1 => 6 - cmp::min(self.stones1.count_ones(), 6),
            ChongPlayer::Player2 => 7 - cmp::min(self.stones2.count_ones(), 7)
        }
    }

    fn coordinate_mask(r: u8, c: u8) -> u64 {
        if r >= 8 || c >= 8 { panic!("The row or column must be between 0 and 7."); }
        1 << (8 * r + c)
    }

    fn pawn_mask(&self) -> u64 {
        let occupied = self.pawn1 | self.pawn2 | self.stones1 | self.stones2;

        let (pawn, stones) = match self.next {
            ChongPlayer::Player1 => (self.pawn1, self.stones1),
            ChongPlayer::Player2 => (self.pawn2, self.stones2)
        };

        ((pawn >> 8) |
         (pawn << 8) |
         ((pawn & 0xfefe_fefe_fefe_fefe) >> 1) |
         ((pawn & 0x7f7f_7f7f_7f7f_7f7f) << 1) |
         (((pawn >> 8) & stones) >> 8) |
         (((pawn << 8) & stones) << 8) |
         ((((pawn & 0xfcfc_fcfc_fcfc_fcfc) >> 1) & stones) >> 1) |
         ((((pawn & 0xfcfc_fcfc_fcfc_fcfc) << 7) & stones) << 7) |
         ((((pawn & 0xfcfc_fcfc_fcfc_fcfc) >> 9) & stones) >> 9) |
         ((((pawn & 0x3f3f_3f3f_3f3f_3f3f) << 1) & stones) << 1) |
         ((((pawn & 0x3f3f_3f3f_3f3f_3f3f) >> 7) & stones) >> 7) |
         ((((pawn & 0x3f3f_3f3f_3f3f_3f3f) << 9) & stones) << 9)) & !occupied
    }

    fn build_state(pawn1: (u8, u8), pawn2: (u8, u8),
                   stones1: &[(u8, u8)], stones2: &[(u8, u8)], next: u8) -> Self {
        let pawn1 = ChongState::coordinate_mask(pawn1.0, pawn1.1);
        let pawn2 = ChongState::coordinate_mask(pawn2.0, pawn2.1);
        let stones1 = stones1.iter()
            .map(|(r, c)| ChongState::coordinate_mask(*r, *c))
            .fold(0, |acc, x| acc | x);
        let stones2 = stones2.iter()
            .map(|(r, c)| ChongState::coordinate_mask(*r, *c))
            .fold(0, |acc, x| acc | x);

        let next = match next {
            1 => ChongPlayer::Player1,
            2 => ChongPlayer::Player2,
            _ => panic!("The player to move can only be 1 or 2.")
        };

        Self {
            pawn1,
            pawn2,
            stones1,
            stones2,
            next
        }
    }
}


impl BoardState<ChongAction, ChongPlayer> for ChongState {
    fn starting_state() -> Self {
        Self {
            pawn1: Self::coordinate_mask(0, 3),
            pawn2: Self::coordinate_mask(7, 4),
            stones1: 0,
            stones2: 0,
            next: ChongPlayer::Player1
        }
    }

    fn previous_player(&self) -> ChongPlayer {
        self.next.next_player()
    }

    fn current_player(&self) -> ChongPlayer {
        self.next
    }

    fn next_state(&self, action: &ChongAction) -> Self {
        let player = self.current_player();
        let value = Self::coordinate_mask(action.r, action.c);
        match action {
            ChongAction { piece: ChongPiece::Pawn, .. } =>
                match player {
                    ChongPlayer::Player1 =>
                        Self { pawn1: value, next: self.next.next_player(), ..*self },
                    ChongPlayer::Player2 =>
                        Self { pawn2: value, next: self.next.next_player(), ..*self },
                }
            ChongAction { piece: ChongPiece::Stone, .. } =>
                match player {
                    ChongPlayer::Player1 =>
                        Self { stones1: value, next: self.next.next_player(), ..*self },
                    ChongPlayer::Player2 =>
                        Self { stones2: value, next: self.next.next_player(), ..*self },
                }
        }
    }

    fn is_legal(&self, action: &ChongAction, history: &[ChongState]) -> bool {
        if action.r >= 8 { return false }
        if action.c >= 8 { return false }

        let occupied = self.pawn1 | self.pawn2 | self.stones1 | self.stones2;
        let value = Self::coordinate_mask(action.r, action.c);

        if value & occupied != 0 { return false }

        match action {
            ChongAction { piece: ChongPiece::Pawn, .. } => {
                self.pawn_mask() & value != 0
            },
            ChongAction { piece: ChongPiece::Stone, .. } => {
                if action.r == 0 || action.r == 7 { false }
                else { self.stones_remaining(self.next) != 0 }
            }
        }
    }

    fn legal_actions(&self, history: &[ChongState]) -> Vec<ChongAction> {
        let mut actions = Vec::with_capacity(56);
        let occupied = self.pawn1 | self.pawn2 | self.stones1 | self.stones2;

        let valid_stones = match self.stones_remaining(self.next) {
            0 => 0,
            _ => !occupied & 0x00ff_ffff_ffff_ff00
        };

        let valid_pawns = self.pawn_mask();

        for r in 0..8 {
            for c in 0..8 {
                let mask = ChongState::coordinate_mask(r, c);

                if mask & valid_stones != 0 {
                    actions.push(ChongAction { piece: ChongPiece::Stone, r, c });
                }

                if mask & valid_pawns != 0 {
                    actions.push(ChongAction { piece: ChongPiece::Pawn, r, c });
                }
            }
        }

        actions
    }

    fn is_ended(&self, history: &[ChongState]) -> bool {
        let current_state = match history.last() {
            None => return false,
            Some(x) => x
        };

        if current_state.pawn1 & 0xff00_0000_0000_0000 != 0 { return true }
        if current_state.pawn2 & 0x0000_0000_0000_00ff != 0 { return true }
        if self.legal_actions(history).is_empty() { return true }
        if history.iter().filter(|&s| s == current_state).count() >= 3 { return true }

        false
    }
}


#[derive(Debug, Copy, Clone)]
pub struct Stats {
    value: i64,
    visits: u64,
}


fn run_simulation(state: &ChongState, history: &[ChongState],
                  table: &mut HashMap<ChongState, Stats>) {
    let mut current = state.clone();
    let mut history_copy = history.to_vec();
    let mut visited_states = vec![];

    let mut moves: u32 = 0;
    let mut expand = true;
    let mut rng = thread_rng();
    loop {
        if current.is_ended(&history_copy) { break; }
        if moves > 500 { break; }

        let legal_actions = current.legal_actions(&history_copy);
        let actions_states = legal_actions.into_iter()
            .map(|a| (a, current.next_state(&a)))
            .collect::<Vec<_>>();
        if expand {
            // if not all of the child nodes are present, expand the nodes
            if !actions_states.iter().all(|(_a, s)| table.contains_key(&s)) {
                for (_a, s) in actions_states.iter() {
                    table.entry(*s).or_insert_with(|| Stats { value: 0, visits: 0 });
                }
                expand = false;
            }

            let actions_statistics = actions_states.into_iter()
                .map(|(a, s)| {
                    let e = table.get(&s).unwrap();
                    (a, s, e)
                })
                .collect::<Vec<_>>();

            let mut log_total = actions_statistics.iter()
                .map(|(_a, _s, e)| e.visits as f64)
                .sum::<f64>()
                .ln();
            if log_total.is_infinite() {
                log_total = 0.0_f64;
            }
            let values_actions = actions_statistics
                .into_iter()
                .map(|(a, s, e)| {
                    let v = e.value as f64 / cmp::max(e.visits, 1) as f64 +
                        1.4 * (log_total / cmp::max(e.visits, 1) as f64).sqrt();
                    (a, s, v)
                })
                .collect::<Vec<_>>();
            // choose the move from amongst the subset of actions with maximum UCT value
            let max_value = values_actions
                .iter()
                .fold(f64::NAN, |acc, (_a, _s, v)| acc.max(*v));
            let choices = values_actions
                .into_iter()
                .filter(|x| x.2 == max_value)
                .collect::<Vec<_>>();
            current = rng.choose(&choices).unwrap().1;
            history_copy.push(current);
            visited_states.push(current);
        }
        else {
            // randomly choose the move, perhaps weighted by an algorithm
            let choices = actions_states
                .into_iter()
                .map(|(a, s)| (a, s, 0.0))
                .collect::<Vec<_>>();
            current = rng.choose(&choices).unwrap().1;
            history_copy.push(current);
        }

        moves += 1;
    }
}


fn mcts(current: &ChongState, history: &[ChongState]) -> bool {
    let mut games: u32 = 0;
    let mut table = HashMap::new();
    loop {
        if games > 1_000 { break; }
        run_simulation(&current, &history, &mut table);
        games += 1;
    }

    true
}


fn main() {
    let mut buffer = String::new();

    io::stdin().read_to_string(&mut buffer)
               .expect("Failed to read input.");

    let history: Vec<ChongState> = serde_json::from_str(&buffer).unwrap();
    let current_state = match history.last() {
        None => return,
        Some(x) => x,
    };

    let result = mcts(&current_state, &history);
    println!("result = {}", result);
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pawn_move_center() {
        let position = ChongState::build_state((3, 4), (7, 4), &[], &[], 1);
        let mut valid_moves = Vec::new();

        for r in 0..8 {
            for c in 0..8 {
                let action = ChongAction { piece: ChongPiece::Pawn, r: r, c: c };
                if position.is_legal(&action, &[]) { valid_moves.push((r, c)); }
            }
        }

        assert_eq!(valid_moves, [(2, 4), (3, 3), (3, 5), (4, 4)]);

        let legal_actions = position.legal_actions(&[])
            .into_iter()
            .filter(|a| a.piece == ChongPiece::Pawn)
            .collect::<Vec<ChongAction>>();
        assert_eq!(legal_actions, [
            ChongAction { piece: ChongPiece::Pawn, r: 2, c: 4 },
            ChongAction { piece: ChongPiece::Pawn, r: 3, c: 3 },
            ChongAction { piece: ChongPiece::Pawn, r: 3, c: 5 },
            ChongAction { piece: ChongPiece::Pawn, r: 4, c: 4 }
            ]);

        let resulting_states = legal_actions.iter()
            .map(|a| position.next_state(&a))
            .collect::<Vec<_>>();
        assert_eq!(resulting_states, [
            ChongState { pawn1: ChongState::coordinate_mask(2, 4), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(3, 3), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(3, 5), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(4, 4), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 }
            ]);
    }

    #[test]
    fn pawn_move_upper_edge() {
        let position = ChongState::build_state((0, 3), (7, 4), &[], &[], 1);
        let mut valid_moves = Vec::new();

        for r in 0..8 {
            for c in 0..8 {
                let action = ChongAction { piece: ChongPiece::Pawn, r: r, c: c };
                if position.is_legal(&action, &[]) { valid_moves.push((r, c)); }
            }
        }

        assert_eq!(valid_moves, [(0, 2), (0, 4), (1, 3)]);

        let legal_actions = position.legal_actions(&[])
            .into_iter()
            .filter(|a| a.piece == ChongPiece::Pawn)
            .collect::<Vec<ChongAction>>();
        assert_eq!(legal_actions, [
            ChongAction { piece: ChongPiece::Pawn, r: 0, c: 2 },
            ChongAction { piece: ChongPiece::Pawn, r: 0, c: 4 },
            ChongAction { piece: ChongPiece::Pawn, r: 1, c: 3 }
        ]);

        let resulting_states = legal_actions.iter()
            .map(|a| position.next_state(&a))
            .collect::<Vec<_>>();
        assert_eq!(resulting_states, [
            ChongState { pawn1: ChongState::coordinate_mask(0, 2), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(0, 4), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(1, 3), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 }
            ]);
    }

    #[test]
    fn pawn_move_lower_edge() {
        let position = ChongState::build_state((7, 4), (0, 0), &[], &[], 1);
        let mut valid_moves = Vec::new();

        for r in 0..8 {
            for c in 0..8 {
                let action = ChongAction { piece: ChongPiece::Pawn, r: r, c: c };
                if position.is_legal(&action, &[]) { valid_moves.push((r, c)); }
            }
        }

        assert_eq!(valid_moves, [(6, 4), (7, 3), (7, 5)]);

        let legal_actions = position.legal_actions(&[])
            .into_iter()
            .filter(|a| a.piece == ChongPiece::Pawn)
            .collect::<Vec<ChongAction>>();
        assert_eq!(legal_actions, [
            ChongAction { piece: ChongPiece::Pawn, r: 6, c: 4 },
            ChongAction { piece: ChongPiece::Pawn, r: 7, c: 3 },
            ChongAction { piece: ChongPiece::Pawn, r: 7, c: 5 }
        ]);

        let resulting_states = legal_actions.iter()
            .map(|a| position.next_state(&a))
            .collect::<Vec<_>>();
        assert_eq!(resulting_states, [
            ChongState { pawn1: ChongState::coordinate_mask(6, 4), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(7, 3), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(7, 5), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 }
            ]);
    }

    #[test]
    fn pawn_move_left_edge() {
        let position = ChongState::build_state((2, 0), (7, 7), &[], &[], 1);
        let mut valid_moves = Vec::new();

        for r in 0..8 {
            for c in 0..8 {
                let action = ChongAction { piece: ChongPiece::Pawn, r: r, c: c };
                if position.is_legal(&action, &[]) { valid_moves.push((r, c)); }
            }
        }

        assert_eq!(valid_moves, [(1, 0), (2, 1), (3, 0)]);

        let legal_actions = position.legal_actions(&[])
            .into_iter()
            .filter(|a| a.piece == ChongPiece::Pawn)
            .collect::<Vec<ChongAction>>();
        assert_eq!(legal_actions, [
            ChongAction { piece: ChongPiece::Pawn, r: 1, c: 0 },
            ChongAction { piece: ChongPiece::Pawn, r: 2, c: 1 },
            ChongAction { piece: ChongPiece::Pawn, r: 3, c: 0 }
        ]);

        let resulting_states = legal_actions.iter()
            .map(|a| position.next_state(&a))
            .collect::<Vec<_>>();
        assert_eq!(resulting_states, [
            ChongState { pawn1: ChongState::coordinate_mask(1, 0), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(2, 1), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(3, 0), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 }
            ]);
    }

    #[test]
    fn pawn_move_right_edge() {
        let position = ChongState::build_state((5, 7), (0, 0), &[], &[], 1);
        let mut valid_moves = Vec::new();

        for r in 0..8 {
            for c in 0..8 {
                let action = ChongAction { piece: ChongPiece::Pawn, r: r, c: c };
                if position.is_legal(&action, &[]) { valid_moves.push((r, c)); }
            }
        }

        assert_eq!(valid_moves, [(4, 7), (5, 6), (6, 7)]);

        let legal_actions = position.legal_actions(&[])
            .into_iter()
            .filter(|a| a.piece == ChongPiece::Pawn)
            .collect::<Vec<ChongAction>>();
        assert_eq!(legal_actions, [
            ChongAction { piece: ChongPiece::Pawn, r: 4, c: 7 },
            ChongAction { piece: ChongPiece::Pawn, r: 5, c: 6 },
            ChongAction { piece: ChongPiece::Pawn, r: 6, c: 7 }
        ]);

        let resulting_states = legal_actions.iter()
            .map(|a| position.next_state(&a))
            .collect::<Vec<_>>();
        assert_eq!(resulting_states, [
            ChongState { pawn1: ChongState::coordinate_mask(4, 7), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(5, 6), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(6, 7), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 }
            ]);
    }

    #[test]
    fn pawn_move_top_left() {
        let position = ChongState::build_state((0, 0), (5, 5), &[], &[], 1);
        let mut valid_moves = Vec::new();

        for r in 0..8 {
            for c in 0..8 {
                let action = ChongAction { piece: ChongPiece::Pawn, r: r, c: c };
                if position.is_legal(&action, &[]) { valid_moves.push((r, c)); }
            }
        }

        assert_eq!(valid_moves, [(0, 1), (1, 0)]);

        let legal_actions = position.legal_actions(&[])
            .into_iter()
            .filter(|a| a.piece == ChongPiece::Pawn)
            .collect::<Vec<ChongAction>>();
        assert_eq!(legal_actions, [
            ChongAction { piece: ChongPiece::Pawn, r: 0, c: 1 },
            ChongAction { piece: ChongPiece::Pawn, r: 1, c: 0 }
        ]);

        let resulting_states = legal_actions.iter()
            .map(|a| position.next_state(&a))
            .collect::<Vec<_>>();
        assert_eq!(resulting_states, [
            ChongState { pawn1: ChongState::coordinate_mask(0, 1), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(1, 0), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 }
            ]);
    }

    #[test]
    fn pawn_move_top_right() {
        let position = ChongState::build_state((0, 7), (5, 5), &[], &[], 1);
        let mut valid_moves = Vec::new();

        for r in 0..8 {
            for c in 0..8 {
                let action = ChongAction { piece: ChongPiece::Pawn, r: r, c: c };
                if position.is_legal(&action, &[]) { valid_moves.push((r, c)); }
            }
        }

        assert_eq!(valid_moves, [(0, 6), (1, 7)]);

        let legal_actions = position.legal_actions(&[])
            .into_iter()
            .filter(|a| a.piece == ChongPiece::Pawn)
            .collect::<Vec<ChongAction>>();
        assert_eq!(legal_actions, [
            ChongAction { piece: ChongPiece::Pawn, r: 0, c: 6 },
            ChongAction { piece: ChongPiece::Pawn, r: 1, c: 7 }
        ]);

        let resulting_states = legal_actions.iter()
            .map(|a| position.next_state(&a))
            .collect::<Vec<_>>();
        assert_eq!(resulting_states, [
            ChongState { pawn1: ChongState::coordinate_mask(0, 6), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(1, 7), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 }
            ]);
    }

    #[test]
    fn pawn_move_bottom_left() {
        let position = ChongState::build_state((7, 0), (5, 5), &[], &[], 1);
        let mut valid_moves = Vec::new();

        for r in 0..8 {
            for c in 0..8 {
                let action = ChongAction { piece: ChongPiece::Pawn, r: r, c: c };
                if position.is_legal(&action, &[]) { valid_moves.push((r, c)); }
            }
        }

        assert_eq!(valid_moves, [(6, 0), (7, 1)]);

        let legal_actions = position.legal_actions(&[])
            .into_iter()
            .filter(|a| a.piece == ChongPiece::Pawn)
            .collect::<Vec<ChongAction>>();
        assert_eq!(legal_actions, [
            ChongAction { piece: ChongPiece::Pawn, r: 6, c: 0 },
            ChongAction { piece: ChongPiece::Pawn, r: 7, c: 1 }
        ]);

        let resulting_states = legal_actions.iter()
            .map(|a| position.next_state(&a))
            .collect::<Vec<_>>();
        assert_eq!(resulting_states, [
            ChongState { pawn1: ChongState::coordinate_mask(6, 0), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(7, 1), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 }
            ]);
    }

    #[test]
    fn pawn_move_bottom_right() {
        let position = ChongState::build_state((7, 7), (5, 5), &[], &[], 1);
        let mut valid_moves = Vec::new();

        for r in 0..8 {
            for c in 0..8 {
                let action = ChongAction { piece: ChongPiece::Pawn, r: r, c: c };
                if position.is_legal(&action, &[]) { valid_moves.push((r, c)); }
            }
        }

        assert_eq!(valid_moves, [(6, 7), (7, 6)]);

        let legal_actions = position.legal_actions(&[])
            .into_iter()
            .filter(|a| a.piece == ChongPiece::Pawn)
            .collect::<Vec<ChongAction>>();
        assert_eq!(legal_actions, [
            ChongAction { piece: ChongPiece::Pawn, r: 6, c: 7 },
            ChongAction { piece: ChongPiece::Pawn, r: 7, c: 6 }
        ]);

        let resulting_states = legal_actions.iter()
            .map(|a| position.next_state(&a))
            .collect::<Vec<_>>();
        assert_eq!(resulting_states, [
            ChongState { pawn1: ChongState::coordinate_mask(6, 7), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(7, 6), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 }
            ]);
    }

    #[test]
    fn pawn_jump_center() {
        let stones = [(2, 3), (2, 4), (2, 5), (3, 3), (3, 5), (4, 3), (4, 4), (4, 5)];
        let position = ChongState::build_state((3, 4), (7, 7), &stones, &[], 1);
        let mut valid_moves = Vec::new();

        for r in 0..8 {
            for c in 0..8 {
                let action = ChongAction { piece: ChongPiece::Pawn, r: r, c: c };
                if position.is_legal(&action, &[]) { valid_moves.push((r, c)); }
            }
        }

        assert_eq!(valid_moves, [(1, 2), (1, 4), (1, 6), (3, 2), (3, 6), (5, 2), (5, 4), (5, 6)]);

        let legal_actions = position.legal_actions(&[])
            .into_iter()
            .filter(|a| a.piece == ChongPiece::Pawn)
            .collect::<Vec<ChongAction>>();
        assert_eq!(legal_actions, [
            ChongAction { piece: ChongPiece::Pawn, r: 1, c: 2 },
            ChongAction { piece: ChongPiece::Pawn, r: 1, c: 4 },
            ChongAction { piece: ChongPiece::Pawn, r: 1, c: 6 },
            ChongAction { piece: ChongPiece::Pawn, r: 3, c: 2 },
            ChongAction { piece: ChongPiece::Pawn, r: 3, c: 6 },
            ChongAction { piece: ChongPiece::Pawn, r: 5, c: 2 },
            ChongAction { piece: ChongPiece::Pawn, r: 5, c: 4 },
            ChongAction { piece: ChongPiece::Pawn, r: 5, c: 6 }
        ]);

        let resulting_states = legal_actions.iter()
            .map(|a| position.next_state(&a))
            .collect::<Vec<_>>();
        assert_eq!(resulting_states, [
            ChongState { pawn1: ChongState::coordinate_mask(1, 2), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(1, 4), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(1, 6), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(3, 2), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(3, 6), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(5, 2), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(5, 4), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(5, 6), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 }
            ]);
    }

    #[test]
    fn pawn_jump_top_edge() {
        let stones = [(0, 2), (0, 4), (1, 2), (1, 3), (1, 4), (7, 2), (7, 3), (7, 4)];
        let position = ChongState::build_state((0, 3), (7, 7), &stones, &[], 1);
        let mut valid_moves = Vec::new();

        for r in 0..8 {
            for c in 0..8 {
                let action = ChongAction { piece: ChongPiece::Pawn, r: r, c: c };
                if position.is_legal(&action, &[]) { valid_moves.push((r, c)); }
            }
        }

        assert_eq!(valid_moves, [(0, 1), (0, 5), (2, 1), (2, 3), (2, 5)]);

        let legal_actions = position.legal_actions(&[])
            .into_iter()
            .filter(|a| a.piece == ChongPiece::Pawn)
            .collect::<Vec<ChongAction>>();
        assert_eq!(legal_actions, [
            ChongAction { piece: ChongPiece::Pawn, r: 0, c: 1 },
            ChongAction { piece: ChongPiece::Pawn, r: 0, c: 5 },
            ChongAction { piece: ChongPiece::Pawn, r: 2, c: 1 },
            ChongAction { piece: ChongPiece::Pawn, r: 2, c: 3 },
            ChongAction { piece: ChongPiece::Pawn, r: 2, c: 5 }
        ]);

        let resulting_states = legal_actions.iter()
            .map(|a| position.next_state(&a))
            .collect::<Vec<_>>();
        assert_eq!(resulting_states, [
            ChongState { pawn1: ChongState::coordinate_mask(0, 1), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(0, 5), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(2, 1), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(2, 3), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(2, 5), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 }
            ]);
    }

    #[test]
    fn pawn_jump_bottom_edge() {
        let stones = [(7, 3), (7, 5), (6, 3), (6, 4), (6, 5), (0, 3), (0, 4), (0, 5)];
        let position = ChongState::build_state((7, 4), (0, 0), &stones, &[], 1);
        let mut valid_moves = Vec::new();

        for r in 0..8 {
            for c in 0..8 {
                let action = ChongAction { piece: ChongPiece::Pawn, r: r, c: c };
                if position.is_legal(&action, &[]) { valid_moves.push((r, c)); }
            }
        }

        assert_eq!(valid_moves, [(5, 2), (5, 4), (5, 6), (7, 2), (7, 6)]);

        let legal_actions = position.legal_actions(&[])
            .into_iter()
            .filter(|a| a.piece == ChongPiece::Pawn)
            .collect::<Vec<ChongAction>>();
        assert_eq!(legal_actions, [
            ChongAction { piece: ChongPiece::Pawn, r: 5, c: 2 },
            ChongAction { piece: ChongPiece::Pawn, r: 5, c: 4 },
            ChongAction { piece: ChongPiece::Pawn, r: 5, c: 6 },
            ChongAction { piece: ChongPiece::Pawn, r: 7, c: 2 },
            ChongAction { piece: ChongPiece::Pawn, r: 7, c: 6 }
        ]);

        let resulting_states = legal_actions.iter()
            .map(|a| position.next_state(&a))
            .collect::<Vec<_>>();
        assert_eq!(resulting_states, [
            ChongState { pawn1: ChongState::coordinate_mask(5, 2), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(5, 4), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(5, 6), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(7, 2), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(7, 6), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 }
            ]);
    }

    #[test]
    fn pawn_jump_left_edge() {
        let stones = [(2, 0), (2, 1), (3, 1), (4, 0), (4, 1), (1, 7), (2, 7), (3, 7), (4, 7)];
        let position = ChongState::build_state((3, 0), (7, 7), &stones, &[], 1);
        let mut valid_moves = Vec::new();

        for r in 0..8 {
            for c in 0..8 {
                let action = ChongAction { piece: ChongPiece::Pawn, r: r, c: c };
                if position.is_legal(&action, &[]) { valid_moves.push((r, c)); }
            }
        }

        assert_eq!(valid_moves, [(1, 0), (1, 2), (3, 2), (5, 0), (5, 2)]);

        let legal_actions = position.legal_actions(&[])
            .into_iter()
            .filter(|a| a.piece == ChongPiece::Pawn)
            .collect::<Vec<ChongAction>>();
        assert_eq!(legal_actions, [
            ChongAction { piece: ChongPiece::Pawn, r: 1, c: 0 },
            ChongAction { piece: ChongPiece::Pawn, r: 1, c: 2 },
            ChongAction { piece: ChongPiece::Pawn, r: 3, c: 2 },
            ChongAction { piece: ChongPiece::Pawn, r: 5, c: 0 },
            ChongAction { piece: ChongPiece::Pawn, r: 5, c: 2 }
        ]);

        let resulting_states = legal_actions.iter()
            .map(|a| position.next_state(&a))
            .collect::<Vec<_>>();
        assert_eq!(resulting_states, [
            ChongState { pawn1: ChongState::coordinate_mask(1, 0), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(1, 2), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(3, 2), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(5, 0), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(5, 2), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 }
            ]);
    }

    #[test]
    fn pawn_jump_right_edge() {
        let stones = [(3, 6), (3, 7), (4, 6), (5, 6), (5, 7), (3, 0), (4, 0), (5, 0), (6, 0)];
        let position = ChongState::build_state((4, 7), (0, 0), &stones, &[], 1);
        let mut valid_moves = Vec::new();

        for r in 0..8 {
            for c in 0..8 {
                let action = ChongAction { piece: ChongPiece::Pawn, r: r, c: c };
                if position.is_legal(&action, &[]) { valid_moves.push((r, c)); }
            }
        }

        assert_eq!(valid_moves, [(2, 5), (2, 7), (4, 5), (6, 5), (6, 7)]);

        let legal_actions = position.legal_actions(&[])
            .into_iter()
            .filter(|a| a.piece == ChongPiece::Pawn)
            .collect::<Vec<ChongAction>>();
        assert_eq!(legal_actions, [
            ChongAction { piece: ChongPiece::Pawn, r: 2, c: 5 },
            ChongAction { piece: ChongPiece::Pawn, r: 2, c: 7 },
            ChongAction { piece: ChongPiece::Pawn, r: 4, c: 5 },
            ChongAction { piece: ChongPiece::Pawn, r: 6, c: 5 },
            ChongAction { piece: ChongPiece::Pawn, r: 6, c: 7 }
        ]);

        let resulting_states = legal_actions.iter()
            .map(|a| position.next_state(&a))
            .collect::<Vec<_>>();
        assert_eq!(resulting_states, [
            ChongState { pawn1: ChongState::coordinate_mask(2, 5), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(2, 7), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(4, 5), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(6, 5), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(6, 7), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 }
            ]);
    }

    #[test]
    fn pawn_jump_top_left() {
        let stones = [(0, 1), (1, 0), (1, 1), (0, 7), (1, 7), (2, 7), (7, 0), (7, 1), (7, 7)];
        let position = ChongState::build_state((0, 0), (5, 5), &stones, &[], 1);
        let mut valid_moves = Vec::new();

        for r in 0..8 {
            for c in 0..8 {
                let action = ChongAction { piece: ChongPiece::Pawn, r: r, c: c };
                if position.is_legal(&action, &[]) { valid_moves.push((r, c)); }
            }
        }

        assert_eq!(valid_moves, [(0, 2), (2, 0), (2, 2)]);

        let legal_actions = position.legal_actions(&[])
            .into_iter()
            .filter(|a| a.piece == ChongPiece::Pawn)
            .collect::<Vec<ChongAction>>();
        assert_eq!(legal_actions, [
            ChongAction { piece: ChongPiece::Pawn, r: 0, c: 2 },
            ChongAction { piece: ChongPiece::Pawn, r: 2, c: 0 },
            ChongAction { piece: ChongPiece::Pawn, r: 2, c: 2 }
        ]);

        let resulting_states = legal_actions.iter()
            .map(|a| position.next_state(&a))
            .collect::<Vec<_>>();
        assert_eq!(resulting_states, [
            ChongState { pawn1: ChongState::coordinate_mask(0, 2), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(2, 0), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(2, 2), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 }
            ]);
    }

    #[test]
    fn pawn_jump_top_right() {
        let stones = [(0, 6), (1, 6), (1, 7), (0, 0), (1, 0), (2, 0), (7, 6), (7, 7), (7, 0)];
        let position = ChongState::build_state((0, 7), (5, 5), &stones, &[], 1);
        let mut valid_moves = Vec::new();

        for r in 0..8 {
            for c in 0..8 {
                let action = ChongAction { piece: ChongPiece::Pawn, r: r, c: c };
                if position.is_legal(&action, &[]) { valid_moves.push((r, c)); }
            }
        }

        assert_eq!(valid_moves, [(0, 5), (2, 5), (2, 7)]);

        let legal_actions = position.legal_actions(&[])
            .into_iter()
            .filter(|a| a.piece == ChongPiece::Pawn)
            .collect::<Vec<ChongAction>>();
        assert_eq!(legal_actions, [
            ChongAction { piece: ChongPiece::Pawn, r: 0, c: 5 },
            ChongAction { piece: ChongPiece::Pawn, r: 2, c: 5 },
            ChongAction { piece: ChongPiece::Pawn, r: 2, c: 7 }
        ]);

        let resulting_states = legal_actions.iter()
            .map(|a| position.next_state(&a))
            .collect::<Vec<_>>();
        assert_eq!(resulting_states, [
            ChongState { pawn1: ChongState::coordinate_mask(0, 5), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(2, 5), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(2, 7), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 }
            ]);
    }

    #[test]
    fn pawn_jump_bottom_left() {
        let stones = [(6, 0), (6, 1), (7, 1), (7, 7), (6, 7), (5, 7)];
        let position = ChongState::build_state((7, 0), (4, 4), &stones, &[], 1);
        let mut valid_moves = Vec::new();

        for r in 0..8 {
            for c in 0..8 {
                let action = ChongAction { piece: ChongPiece::Pawn, r: r, c: c };
                if position.is_legal(&action, &[]) { valid_moves.push((r, c)); }
            }
        }

        assert_eq!(valid_moves, [(5, 0), (5, 2), (7, 2)]);

        let legal_actions = position.legal_actions(&[])
            .into_iter()
            .filter(|a| a.piece == ChongPiece::Pawn)
            .collect::<Vec<ChongAction>>();
        assert_eq!(legal_actions, [
            ChongAction { piece: ChongPiece::Pawn, r: 5, c: 0 },
            ChongAction { piece: ChongPiece::Pawn, r: 5, c: 2 },
            ChongAction { piece: ChongPiece::Pawn, r: 7, c: 2 }
        ]);

        let resulting_states = legal_actions.iter()
            .map(|a| position.next_state(&a))
            .collect::<Vec<_>>();
        assert_eq!(resulting_states, [
            ChongState { pawn1: ChongState::coordinate_mask(5, 0), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(5, 2), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(7, 2), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 }
            ]);
    }

    #[test]
    fn pawn_jump_bottom_right() {
        let stones = [(6, 6), (6, 7), (7, 6), (7, 1), (0, 1), (0, 6), (0, 7), (1, 0)];
        let position = ChongState::build_state((7, 7), (4, 4), &stones, &[], 1);
        let mut valid_moves = Vec::new();

        for r in 0..8 {
            for c in 0..8 {
                let action = ChongAction { piece: ChongPiece::Pawn, r: r, c: c };
                if position.is_legal(&action, &[]) { valid_moves.push((r, c)); }
            }
        }

        assert_eq!(valid_moves, [(5, 5), (5, 7), (7, 5)]);

        let legal_actions = position.legal_actions(&[])
            .into_iter()
            .filter(|a| a.piece == ChongPiece::Pawn)
            .collect::<Vec<ChongAction>>();
        assert_eq!(legal_actions, [
            ChongAction { piece: ChongPiece::Pawn, r: 5, c: 5 },
            ChongAction { piece: ChongPiece::Pawn, r: 5, c: 7 },
            ChongAction { piece: ChongPiece::Pawn, r: 7, c: 5 }
        ]);

        let resulting_states = legal_actions.iter()
            .map(|a| position.next_state(&a))
            .collect::<Vec<_>>();
        assert_eq!(resulting_states, [
            ChongState { pawn1: ChongState::coordinate_mask(5, 5), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(5, 7), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 },
            ChongState { pawn1: ChongState::coordinate_mask(7, 5), pawn2: position.pawn2,
                         stones1: position.stones1, stones2: position.stones2, next: ChongPlayer::Player2 }
            ]);
    }
}
