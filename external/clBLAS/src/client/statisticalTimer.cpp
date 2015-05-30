/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/


// StatTimer.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include <iostream>
#include <string>
#include <cassert>
#include <limits>
#include <functional>
#include "statisticalTimer.h"

#if defined( __GNUC__ )
	#include <sys/time.h>
#endif

//	Functor object to help with accumulating values in vectors
template< typename T >
struct Accumulator: public std::unary_function< T, void >
{
	T acc;

	Accumulator( ): acc( 0 ) {}
	void operator( )(T x) { acc += x; }
};

//	Unary predicate used for remove_if() algorithm
//	Currently, RangeType is expected to be a floating point type, and ValType an integer type
template< typename RangeType, typename ValType >
struct PruneRange
{
	RangeType lower, upper;

	PruneRange( RangeType mean, RangeType stdev ): lower( mean-stdev ), upper( mean+stdev ) {}

	bool operator( )( ValType val )
	{
		//	These comparisons can be susceptible to signed/unsigned casting problems
		//	This is why we cast ValType to RangeType, because RangeType should always be floating and signed
		if( static_cast< RangeType >( val ) < lower )
			return true;
		else if( static_cast< RangeType >( val ) > upper )
			return true;

		return false;
	}
};

StatisticalTimer&
StatisticalTimer::getInstance( )
{
	static	StatisticalTimer	timer;
	return	timer;
}

StatisticalTimer::StatisticalTimer( ): nEvents( 0 ), nSamples( 0 ), normalize( true )
{
#if defined( _WIN32 )
	//	OS call to get ticks per second2
	::QueryPerformanceFrequency( reinterpret_cast<LARGE_INTEGER*>( &clkFrequency ) );
#else
	clkFrequency = 1000000;
#endif
}

StatisticalTimer::~StatisticalTimer( )
{}

void
StatisticalTimer::Clear( )
{
	labelID.clear( );
	clkStart.clear( );
	clkTicks.clear( );
}

void
StatisticalTimer::Reset( )
{
	if( nEvents == 0 || nSamples == 0 )
		throw	std::runtime_error( "StatisticalTimer::Reserve( ) was not called before Reset( )" );

	clkStart.clear( );
	clkTicks.clear( );

	clkStart.resize( nEvents );
	clkTicks.resize( nEvents );

	for( unsigned int	i = 0; i < nEvents; ++i )
	{
		clkTicks.at( i ).reserve( nSamples );
	}

	return;
}

//	The caller can pre-allocate memory, to improve performance.
//	nEvents is an approximate value for how many seperate events the caller will think
//	they will need, and nSamples is a hint on how many samples we think we will take
//	per event
void
StatisticalTimer::Reserve( unsigned int nEvents, unsigned int nSamples )
{
	this->nEvents	= std::max<unsigned int> (1, nEvents);
	this->nSamples	= std::max<unsigned int> (1, nSamples);

	Clear( );
	labelID.reserve( nEvents );

	clkStart.resize( nEvents );
	clkTicks.resize( nEvents );

	for( unsigned int i = 0; i < nEvents; ++i )
	{
		clkTicks.at( i ).reserve( nSamples );
	}
}

void
StatisticalTimer::setNormalize( bool norm )
{
	normalize = norm;
}

void
StatisticalTimer::Start( sTimerID id )
{
#if defined( _WIN32 )
	::QueryPerformanceCounter( reinterpret_cast<LARGE_INTEGER*>( &clkStart.at( id ) ) );
#else
	struct timeval s;
	gettimeofday(&s, 0);
	clkStart.at( id ) = (unsigned long long)s.tv_sec * 1000000 + (unsigned long long)s.tv_usec;
#endif
}

void
StatisticalTimer::Stop( sTimerID id )
{
	unsigned long long n;

#if defined( _WIN32 )
	::QueryPerformanceCounter( reinterpret_cast<LARGE_INTEGER*>( &n ) );
#else
	struct timeval s;
	gettimeofday(&s, 0);
	n = (unsigned long long)s.tv_sec * 1000000 + (unsigned long long)s.tv_usec;
#endif

	n		-= clkStart.at( id );
	clkStart.at( id )	= 0;
	AddSample( id, n );
}

void
StatisticalTimer::AddSample( const sTimerID id, const unsigned long long n )
{
	clkTicks.at( id ).push_back( n );
}

//	This function's purpose is to provide a mapping from a 'friendly' human readable text string
//	to an index into internal data structures.
StatisticalTimer::sTimerID
StatisticalTimer::getUniqueID( const std::string& label, unsigned int groupID )
{
	//	I expect labelID will hardly ever grow beyond 30, so it's not of any use
	//	to keep this sorted and do a binary search

	labelPair	sItem	= std::make_pair( label, groupID );

	stringVector::iterator	iter;
	iter	= std::find( labelID.begin(), labelID.end(), sItem );

	if( iter != labelID.end( ) )
		return	std::distance( labelID.begin( ), iter );

	labelID.push_back( sItem );

	return	labelID.size( ) - 1;

}

double
StatisticalTimer::getMean( sTimerID id ) const
{
	if( clkTicks.empty( ) )
		return	0;

	size_t	N	= clkTicks.at( id ).size( );

	Accumulator<unsigned long long> sum = std::for_each( clkTicks.at( id ).begin(), clkTicks.at( id ).end(), Accumulator<unsigned long long>() );

	return	static_cast<double>( sum.acc ) / N;
}

double
StatisticalTimer::getVariance( sTimerID id ) const
{
	if( clkTicks.empty( ) )
		return	0;

	double	mean	= getMean( id );

	size_t	N	= clkTicks.at( id ).size( );
	double	sum	= 0;

	for( unsigned int i = 0; i < N; ++i )
	{
		double	diff	= clkTicks.at( id ).at( i ) - mean;
		diff	*= diff;
		sum		+= diff;
	}

	return	 sum / N;
}

double
StatisticalTimer::getStdDev( sTimerID id ) const
{
	double	variance	= getVariance( id );

	return	sqrt( variance );
}

double
StatisticalTimer::getAverageTime( sTimerID id ) const
{
	if( normalize )
		return getMean( id ) / clkFrequency;
	else
		return getMean( id );
}

double
StatisticalTimer::getMinimumTime( sTimerID id ) const
{
	clkVector::const_iterator iter	= std::min_element( clkTicks.at( id ).begin( ), clkTicks.at( id ).end( ) );

	if( iter != clkTicks.at( id ).end( ) )
	{
		if( normalize )
			return static_cast<double>( *iter ) / clkFrequency;
		else
			return static_cast<double>( *iter );
	}
	else
		return	0;
}

unsigned int
StatisticalTimer::pruneOutliers( sTimerID id , double multiple )
{
	if( clkTicks.empty( ) )
		return	0;

	double	mean	= getMean( id );
	double	stdDev	= getStdDev( id );

	clkVector&	clks = clkTicks.at( id );

	//	Look on p. 379, "The C++ Standard Library"
	//	std::remove_if does not actually erase, it only copies elements, it returns new 'logical' end
	clkVector::iterator	newEnd	= std::remove_if( clks.begin( ), clks.end( ), PruneRange< double,unsigned long long >( mean, multiple*stdDev ) );

	clkVector::difference_type dist	= std::distance( newEnd, clks.end( ) );

	if( dist != 0 )
		clks.erase( newEnd, clks.end( ) );

	assert( dist < std::numeric_limits< unsigned int >::max( ) );

	return static_cast< unsigned int >( dist );
}

unsigned int
StatisticalTimer::pruneOutliers( double multiple )
{
	unsigned int	tCount	= 0;

	for( unsigned int l = 0; l < labelID.size( ); ++l )
	{
		unsigned int lCount	= pruneOutliers( l , multiple );
		std::clog << "\tStatisticalTimer:: Pruning " << lCount << " samples from " << labelID[l].first << std::endl;
		tCount += lCount;
	}

	return	tCount;
}

//	Defining an output print operator
std::ostream&
operator<<( std::ostream& os, const StatisticalTimer& st )
{
	if( st.clkTicks.empty( ) )
		return	os;

	std::ios::fmtflags bckup	= os.flags( );

	for( unsigned int l = 0; l < st.labelID.size( ); ++l )
	{
		unsigned long long min	= 0;
		StatisticalTimer::clkVector::const_iterator iter	= std::min_element( st.clkTicks.at( l ).begin( ), st.clkTicks.at( l ).end( ) );

		if( iter != st.clkTicks.at( l ).end( ) )
			min		= *iter;

		os << st.labelID[l].first << ", " << st.labelID[l].second << std::fixed << std::endl;
		os << "Min:," << min << std::endl;
		os << "Mean:," << st.getMean( l ) << std::endl;
		os << "StdDev:," << st.getStdDev( l ) << std::endl;
		os << "AvgTime:," << st.getAverageTime( l ) << std::endl;
		os << "MinTime:," << st.getMinimumTime( l ) << std::endl;

		for( unsigned int	t = 0; t < st.clkTicks[l].size( ); ++t )
		{
			os << st.clkTicks[l][t]<< ",";
		}
		os << "\n" << std::endl;

	}

	os.flags( bckup );

	return	os;
}
