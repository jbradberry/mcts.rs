

pub trait BoardAction {}

pub trait BoardState<T: BoardAction> {
    fn starting_state() -> Self;

    fn previous_player(&self) -> usize;

    fn current_player(&self) -> usize;

    fn next_state(&self, action: &T) -> Self;

    // fn is_legal(&self, action: &T, history: &[Self]) -> bool;

    // fn legal_actions(&self) -> Vec<T>;

    // fn is_ended(&self) -> bool;
}


#[derive(Debug)]
pub struct ChongState {
    pawn1: u64,
    pawn2: u64,
    stones1: u64,
    stones2: u64,
    next: usize
}


#[derive(Debug)]
pub enum ChongPiece {
    Pawn,
    Stone
}


#[derive(Debug)]
pub struct ChongAction {
    piece: ChongPiece,
    x: u8,
    y: u8
}


impl BoardAction for ChongAction {}


impl BoardState<ChongAction> for ChongState {
    fn starting_state() -> Self {
        Self {
            pawn1: 1 << (0 * 8 + 3),
            pawn2: 1 << (7 * 8 + 4),
            stones1: 0,
            stones2: 0,
            next: 1
        }
    }

    fn previous_player(&self) -> usize {
        3 - self.next
    }

    fn current_player(&self) -> usize {
        self.next
    }

    fn next_state(&self, action: &ChongAction) -> Self {
        let player = self.current_player();
        let value = 1 << (action.y * 8 + action.x);
        match (action, player) {
            (ChongAction { piece: ChongPiece::Pawn, .. }, 1) =>
                Self { pawn1: value, next: 3 - player, ..*self },
            (ChongAction { piece: ChongPiece::Pawn, .. }, 2) =>
                Self { pawn2: value, next: 3 - player, ..*self },
            (ChongAction { piece: ChongPiece::Stone, .. }, 1) =>
                Self { stones1: value, next: 3 - player, ..*self },
            (ChongAction { piece: ChongPiece::Stone, .. }, 2) =>
                Self { stones2: value, next: 3 - player, ..*self },
            _ =>
                panic!("Something bad happened!")
        }
    }
}


fn main() {
    let start = ChongState::starting_state();
    println!("{:?}", start);
    println!("{}", start.current_player());
    println!("{}", start.previous_player());
    let action = ChongAction { piece: ChongPiece::Pawn, x: 1, y: 3 };
    println!("{:?}", start.next_state(&action))
}
