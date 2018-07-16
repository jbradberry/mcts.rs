use std::str;


pub trait BoardPlayer {}


pub trait BoardAction {}


pub trait BoardState<A: BoardAction, P: BoardPlayer> {
    fn starting_state() -> Self;

    fn previous_player(&self) -> P;

    fn current_player(&self) -> P;

    fn next_state(&self, action: &A) -> Self;

    fn is_legal(&self, action: &A, history: &[Self]) -> bool where Self: Sized;

    // fn legal_actions(&self) -> Vec<T>;

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


#[derive(Debug)]
pub enum ChongPiece {
    Pawn,
    Stone
}


#[derive(Debug)]
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
    fn coordinate_mask(r: u8, c: u8) -> u64 {
        if r >= 8 || c >= 8 { panic!("The row or column must be between 0 and 7."); }
        1 << (8 * r + c)
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
            pawn1: pawn1,
            pawn2: pawn2,
            stones1: stones1,
            stones2: stones2,
            next: next
        }
    }
}


impl BoardState<ChongAction, ChongPlayer> for ChongState {
    fn starting_state() -> Self {
        Self {
            pawn1: 1 << (0 * 8 + 3),
            pawn2: 1 << (7 * 8 + 4),
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
                let player = self.current_player();
                let (pawn, stones) = match player {
                    ChongPlayer::Player1 => (self.pawn1, self.stones1),
                    ChongPlayer::Player2 => (self.pawn2, self.stones2)
                };

                // println!("{}, {}", pawn, value);
                if (pawn << 8) == value || (pawn >> 8) == value { true }
                else if ((pawn << 1) & 0xfefefefefefefefe) == value { true }
                else if ((pawn >> 1) & 0x7f7f7f7f7f7f7f7f) == value { true }
                else if (pawn << 16) == value && ((pawn << 8) & stones) != 0 { true }
                else if (pawn >> 16) == value && ((pawn >> 8) & stones) != 0 { true }
                else if ((pawn << 2) & 0xfefefefefefefefe) == value && ((pawn << 1) & stones) != 0 { true }
                else if ((pawn >> 2) & 0x7f7f7f7f7f7f7f7f) == value && ((pawn >> 1) & stones) != 0 { true }
                else if ((pawn << 14) & 0xfefefefefefefefe) == value && ((pawn << 7) & stones) != 0 { true }
                else if ((pawn >> 14) & 0x7f7f7f7f7f7f7f7f) == value && ((pawn >> 7) & stones) != 0 { true }
                else if ((pawn << 18) & 0xfefefefefefefefe) == value && ((pawn << 9) & stones) != 0 { true }
                else if ((pawn >> 18) & 0x7f7f7f7f7f7f7f7f) == value && ((pawn >> 9) & stones) != 0 { true }
                else { false }
            },
            ChongAction { piece: ChongPiece::Stone, .. } => {
                if action.r == 0 || action.r == 7 { false }
                else if self.stones_remaining(self.next) == 0 { false }
                else { true }
            }
        }
    }
}


impl ChongState {
    fn stones_remaining(&self, player: ChongPlayer) -> u32 {
        match player {
            ChongPlayer::Player1 => 6 - self.stones1.count_ones(),
            ChongPlayer::Player2 => 7 - self.stones2.count_ones(),
        }
    }
}


fn main() {
    let position = ChongState { pawn1: 1 << (0 * 8 + 3), pawn2: 0,
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
    }
}
