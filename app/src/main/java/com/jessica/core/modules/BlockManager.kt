package com.jessica.core.modules

class BlockManager {

    private val blocks =
        mutableListOf<Block>()


    fun addBlock(block: Block) {
        blocks.add(block)
    }


    fun removeBlock(block: Block) {
        blocks.remove(block)
    }


    fun getBlocks(): List<Block> {
        return blocks
    }

}
