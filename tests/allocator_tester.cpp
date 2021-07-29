// SDK includes
#include <bento_base/security.h>
#include <bento_memory/common.h>
#include <bento_memory/page_allocator.h>
#include <bento_memory/book_allocator.h>

struct TByte4
{
	int data;
};

struct TByte8
{
	double data;
};

struct TByte12
{
	int data[3];
};

struct TByte16
{
	bool data[16];
};

struct TByte32
{
	float data[8];
};

void test_page_allocator()
{
	// Create an allocator that allocates 16 bytes per chunk
	bento::PageAllocator pageAllocator;
	pageAllocator.initialize(16);

	// Tests that the structure size correctly accounts for the allocation size
	{
		TByte4* byte4 = bento::make_new<TByte4>(pageAllocator);
		assert(byte4 != nullptr);
		assert(pageAllocator.usage_flags() == 0x00000001);
		bento::make_delete<TByte4>(pageAllocator, byte4);

		TByte8* byte8 = bento::make_new<TByte8>(pageAllocator);
		assert(byte8 != nullptr);
		assert(pageAllocator.usage_flags() == 0x00000001);
		bento::make_delete<TByte8>(pageAllocator, byte8);

		TByte16* byte16 = bento::make_new<TByte16>(pageAllocator);
		assert(byte16 != nullptr);
		assert(pageAllocator.usage_flags() == 0x00000001);
		bento::make_delete<TByte16>(pageAllocator, byte16);

		TByte32* byte32 = bento::make_new<TByte32>(pageAllocator);
		assert(byte32 == nullptr);
		assert(pageAllocator.usage_flags() == 0x00000000);
		bento::make_delete<TByte32>(pageAllocator, byte32);
	}

	// Tests that freeing a given chunk works correctly and that chunk can be re-attributed
	{
		// Allocate the first three chunks
		TByte16* c0 = bento::make_new<TByte16>(pageAllocator);
		assert(c0 != nullptr);
		assert(pageAllocator.usage_flags() == 0x00000001);
		TByte16* c1 = bento::make_new<TByte16>(pageAllocator);
		assert(c1 != nullptr);
		assert(pageAllocator.usage_flags() == 0x00000003);
		TByte16* c2 = bento::make_new<TByte16>(pageAllocator);
		assert(c2 != nullptr);
		assert(pageAllocator.usage_flags() == 0x00000007);

		// Free the middle chunk
		bento::make_delete<TByte16>(pageAllocator, c1);
		assert(pageAllocator.usage_flags() == 0x00000005);

		// Make sure the new data gets reallocated into the middle chunk
		TByte16* c1_bis = bento::make_new<TByte16>(pageAllocator);
		assert(c1_bis != nullptr);
		assert(pageAllocator.usage_flags() == 0x00000007);

		// Delete all the structures
		bento::make_delete<TByte16>(pageAllocator, c0);
		bento::make_delete<TByte16>(pageAllocator, c1_bis);
		bento::make_delete<TByte16>(pageAllocator, c2);
	}

	// Tests that when the page is full, the allocator returns null
	{
		// Allocate the 64 chunks of the page
		TByte16* c[64];
		for (uint32_t chunkIdx = 0; chunkIdx < 64; ++chunkIdx)
		{
			// Allocate the first three chunks
			c[chunkIdx] = bento::make_new<TByte16>(pageAllocator);
			assert(c[chunkIdx] != nullptr);
		}

		// Make sure the usage flag is full
		assert(pageAllocator.is_full());

		TByte16* tooMuch = bento::make_new<TByte16>(pageAllocator);
		assert(tooMuch == nullptr);
		assert(pageAllocator.is_full());

		// Free all the data (top to bottom this time)
		for (int32_t chunkIdx = 63; chunkIdx >= 0; --chunkIdx)
			bento::make_delete<TByte16>(pageAllocator, c[chunkIdx]);

		// Make sure everything was freed
		assert(pageAllocator.usage_flags() == 0x0000000000);
	}
}

void test_book_allocator()
{
	bento::BookAllocator bookAllocator;
	bookAllocator.initialize(4, 16);

	// Grab the 4 pages for checks
	bento::PageAllocator& page0 = bookAllocator.get_page_allocator(0);
	bento::PageAllocator& page1 = bookAllocator.get_page_allocator(1);
	bento::PageAllocator& page2 = bookAllocator.get_page_allocator(2);
	bento::PageAllocator& page3 = bookAllocator.get_page_allocator(3);

	// Test that the allocations fall into the right page (if can)
	{
		TByte4* byte4 = bento::make_new<TByte4>(bookAllocator);
		assert(byte4 != nullptr);
		assert(page0.usage_flags() == 0x00000001);
		bento::make_delete<TByte4>(bookAllocator, byte4);

		TByte8* byte8 = bento::make_new<TByte8>(bookAllocator);
		assert(byte8 != nullptr);
		assert(page1.usage_flags() == 0x00000001);
		bento::make_delete<TByte8>(bookAllocator, byte8);

		TByte16* byte16 = bento::make_new<TByte16>(bookAllocator);
		assert(byte16 != nullptr);
		assert(page3.usage_flags() == 0x00000001);
		bento::make_delete<TByte16>(bookAllocator, byte16);

		TByte32* byte32 = bento::make_new<TByte32>(bookAllocator);
		assert(byte32 == nullptr);
		assert(page0.usage_flags() == 0x00000000);
		assert(page1.usage_flags() == 0x00000000);
		assert(page2.usage_flags() == 0x00000000);
		assert(page3.usage_flags() == 0x00000000);
	}

	// Test that the allocations fall into the next page even if chunk too big
	{
		// Allocate the 64 chunks of the page
		TByte4* c[64];
		for (uint32_t chunkIdx = 0; chunkIdx < 64; ++chunkIdx)
		{
			// Allocate the first three chunks
			c[chunkIdx] = bento::make_new<TByte4>(bookAllocator);
			assert(c[chunkIdx] != nullptr);
		}
		assert(page0.is_full());

		TByte4* extra = bento::make_new<TByte4>(bookAllocator);
		assert(extra != nullptr);
		assert(page0.is_full());
		assert(page1.usage_flags() == 0x00000001);

		// Free all the data (top to bottom this time)
		for (int32_t chunkIdx = 63; chunkIdx >= 0; --chunkIdx)
			bento::make_delete<TByte4>(bookAllocator, c[chunkIdx]);
		bento::make_delete<TByte4>(bookAllocator, extra);

		// Make sure everything was freed
		assert(page0.usage_flags() == 0x0000000000);
		assert(page1.usage_flags() == 0x0000000000);
		assert(page2.usage_flags() == 0x0000000000);
		assert(page3.usage_flags() == 0x0000000000);
	}

	// Test that if all pages are full allocation fails
	{
		// Allocate the 64 chunks of the page
		TByte4* c4[64];
		for (uint32_t chunkIdx = 0; chunkIdx < 64; ++chunkIdx)
		{
			// Allocate the first three chunks
			c4[chunkIdx] = bento::make_new<TByte4>(bookAllocator);
			assert(c4[chunkIdx] != nullptr);
		}
		assert(page0.is_full());

		// Allocate the 64 chunks of the page
		TByte8* c8[64];
		for (uint32_t chunkIdx = 0; chunkIdx < 64; ++chunkIdx)
		{
			// Allocate the first three chunks
			c8[chunkIdx] = bento::make_new<TByte8>(bookAllocator);
			assert(c8[chunkIdx] != nullptr);
		}
		assert(page1.is_full());

		// Allocate the 64 chunks of the page
		TByte12* c12[64];
		for (uint32_t chunkIdx = 0; chunkIdx < 64; ++chunkIdx)
		{
			// Allocate the first three chunks
			c12[chunkIdx] = bento::make_new<TByte12>(bookAllocator);
			assert(c12[chunkIdx] != nullptr);
		}
		assert(page2.is_full());

		// Allocate the 64 chunks of the page
		TByte16* c16[64];
		for (uint32_t chunkIdx = 0; chunkIdx < 64; ++chunkIdx)
		{
			// Allocate the first three chunks
			c16[chunkIdx] = bento::make_new<TByte16>(bookAllocator);
			assert(c16[chunkIdx] != nullptr);
		}
		assert(page3.is_full());

		// Do the smallest too much allocation
		TByte4* tooMuch = bento::make_new<TByte4>(bookAllocator);
		assert(tooMuch == nullptr);

		// Free all the data (top to bottom this time)
		for (int32_t chunkIdx = 63; chunkIdx >= 0; --chunkIdx)
			bento::make_delete<TByte16>(bookAllocator, c16[chunkIdx]);
		for (int32_t chunkIdx = 63; chunkIdx >= 0; --chunkIdx)
			bento::make_delete<TByte12>(bookAllocator, c12[chunkIdx]);
		for (int32_t chunkIdx = 63; chunkIdx >= 0; --chunkIdx)
			bento::make_delete<TByte8>(bookAllocator, c8[chunkIdx]);
		for (int32_t chunkIdx = 63; chunkIdx >= 0; --chunkIdx)
			bento::make_delete<TByte4>(bookAllocator, c4[chunkIdx]);

		// Make sure everything was freed
		assert(page0.usage_flags() == 0x0000000000);
		assert(page1.usage_flags() == 0x0000000000);
		assert(page2.usage_flags() == 0x0000000000);
		assert(page3.usage_flags() == 0x0000000000);
	}
}

int main()
{
	// Run the page allocator tests
	test_page_allocator();

	// Run the book allocator tests
	test_book_allocator();

	// If we got here, everything is fine
	return 0;
}