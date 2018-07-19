use std::cmp;


pub trait BoardPlayer {}


pub trait BoardAction {}


pub trait BoardState<A: BoardAction, P: BoardPlayer> {
    fn starting_state() -> Self;

    fn previous_player(&self) -> P;

    fn current_player(&self) -> P;

    fn next_state(&self, action: &A) -> Self;

    fn is_legal(&self, action: &A, history: &[Self]) -> bool where Self: Sized;

    fn legal_actions(&self, history: &[Self]) -> Vec<A> where Self: Sized;

    // fn is_ended(&self) -> bool;
}


#[derive(Debug, Clone, Copy)]
pub enum ChongPlayer {
    Player1,
    Player2
}


#[derive(Debug)]
pub struct ChongState {
    pawn1: u64,
    pawn2: u64,
    stones1: u64,
    stones2: u64,
    next: ChongPlayer
}


#[derive(Debug, PartialEq)]
pub enum ChongPiece {
    Pawn,
    Stone
}


#[derive(Debug, PartialEq)]
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
         ((pawn & 0xfefefefefefefefe) >> 1) |
         ((pawn & 0x7f7f7f7f7f7f7f7f) << 1) |
         (((pawn >> 8) & stones) >> 8) |
         (((pawn << 8) & stones) << 8) |
         ((((pawn & 0xfcfcfcfcfcfcfcfc) >> 1) & stones) >> 1) |
         ((((pawn & 0xfcfcfcfcfcfcfcfc) << 7) & stones) << 7) |
         ((((pawn & 0xfcfcfcfcfcfcfcfc) >> 9) & stones) >> 9) |
         ((((pawn & 0x3f3f3f3f3f3f3f3f) << 1) & stones) << 1) |
         ((((pawn & 0x3f3f3f3f3f3f3f3f) >> 7) & stones) >> 7) |
         ((((pawn & 0x3f3f3f3f3f3f3f3f) << 9) & stones) << 9)) & !occupied
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
        let value = 1 << (action.r * 8 + action.c);
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
        let value = 1 << (action.r * 8 + action.c);

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
            _ => !occupied & 0x00ffffffffffff00
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
}


fn main() {
    let _position = ChongState { pawn1: ChongState::coordinate_mask(0, 3), pawn2: 0,
                                 stones1: 0, stones2: 0, next: ChongPlayer::Player1 };

    // println!("{:?}", start.is_legal(&action, &[]));
    // println!("{:?}", start.next_state(&action));
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
    }
}
